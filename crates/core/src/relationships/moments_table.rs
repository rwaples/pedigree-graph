//! The relationship-moments boundary arithmetic every host shares (ADR 0015).
//!
//! The engine pass ([`super::relationship_moments`]) takes packed labels and
//! quantized integers and hands back exact `i128` accumulators.  Everything
//! on either side of it lives here, so Python and R compute it by one
//! implementation: packing named factors into labels ([`pack_labels`]),
//! quantizing value columns ([`quantize_column`]), the exact table and its
//! operations ([`MomentsTable`]: select, sum, merge), and every float the
//! table offers ([`MomentsTable::derive`]), each an exact rational rounded
//! once ([`ratio`]).
//!
//! A table's accumulators are little-endian two's complement integers of
//! one width per table.  A host hands a table in at any width and core
//! borrows its bytes; every table core returns is at the smallest width
//! that holds its values, so a table of small sums costs a few bytes per
//! accumulator, a merge of tables whose exponents lie thousands of bits
//! apart still fits, and one set of values has one encoding.  No operation
//! decodes a whole table: each accumulator is decoded when it is read, so
//! the scratch beyond the input and the output is a few big integers.

use super::moments::{CellLayout, Operand, Product, Side, Slot};
use crate::alloc::{self, Family};
use crate::error::Error;
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::borrow::Cow;
use std::cmp::Ordering;

/// Bits a quantized value may use: `max|x| · 2^e <= 2^43` (ADR 0013).
pub const QUANTIZED_BITS: i64 = 43;

/// The most value columns a moments call takes.
pub const MAX_VALUE_COLUMNS: usize = 32;

/// The most equality keys a moments call takes: each doubles the cells, and
/// sixteen is 65,536 times the label product, past which no cell table fits
/// a sensible budget.
pub const MAX_SAME_KEYS: usize = 16;

/// The largest exponent magnitude a table may carry.  The quantizer's range
/// is `[-981, 1117]`; the bound keeps a hand-built table from asking a
/// merge for an unbounded shift.
pub const MAX_EXPONENT: i64 = 1 << 16;

/// Factors packed by mixed radix into one label per row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedLabels {
    /// One label per row, `0..n_labels`.
    pub labels: Vec<i32>,
    /// The product of the factors' level counts (an empty factor counts 1).
    pub n_labels: usize,
    /// Each factor's distinct values, ascending: the levels of its axis.
    pub levels: Vec<Vec<i64>>,
}

/// Pack `factors` (each one value per row) into one label per row: the
/// first factor is the most significant digit, each digit the position of
/// the row's value among the factor's distinct values.
///
/// # Errors
///
/// [`Error::LengthMismatch`] when a factor does not have `n` rows;
/// [`Error::ValueOutOfRange`] (`field`) when the labels pass the int32
/// range; [`Error::AllocationFailed`] for the labels or levels.
pub fn pack_labels(
    factors: &[&[i64]],
    n: usize,
    field: &'static str,
) -> Result<PackedLabels, Error> {
    let mut labels: Vec<i64> = alloc::filled(0, n, Family::MomentInput)?;
    let mut n_labels: usize = 1;
    let mut levels = Vec::with_capacity(factors.len());
    for column in factors {
        super::check_column_length(field, column.len(), n)?;
        let mut distinct = alloc::with_capacity(n, Family::MomentInput)?;
        distinct.extend_from_slice(column);
        distinct.sort_unstable();
        distinct.dedup();
        distinct.shrink_to_fit();
        let radix = distinct.len().max(1);
        n_labels = n_labels.saturating_mul(radix);
        if n_labels > i32::MAX as usize + 1 {
            return Err(Error::ValueOutOfRange {
                field,
                position: 0,
                value: i64::try_from(n_labels - 1).unwrap_or(i64::MAX),
                minimum: 0,
                maximum: i64::from(i32::MAX),
            });
        }
        for (label, value) in labels.iter_mut().zip(column.iter()) {
            // Every value is among the distinct values it was drawn from.
            let digit = distinct.binary_search(value).unwrap_or(0) as i64;
            *label = *label * distinct.len() as i64 + digit;
        }
        levels.push(distinct);
    }
    let mut packed = alloc::with_capacity(n, Family::MomentInput)?;
    // n_labels <= 2^31, so every label fits.
    packed.extend(labels.iter().map(|&label| label as i32));
    Ok(PackedLabels {
        labels: packed,
        n_labels,
        levels,
    })
}

/// The largest integer `e` with `max|x| · 2^e <= 2^43`, from the binary
/// exponent of `max|x|`; `0` for an all-zero (or empty) column.
fn exponent_of(magnitude: f64) -> i64 {
    if magnitude == 0.0 {
        return 0;
    }
    let (mantissa, exponent) = decompose(magnitude);
    // |x| = m · 2^E with m in [2^52, 2^53) or, subnormal, below; frexp's
    // E is bits(m) + E.  m · 2^(43 - frexp E) < 2^43, and exactly 2^43 is
    // allowed when m is a power of two.
    let bits = i64::from(64 - mantissa.leading_zeros());
    let frexp = bits + exponent;
    QUANTIZED_BITS - frexp + i64::from(mantissa.is_power_of_two())
}

/// A finite `x` as `(m, E)` with `|x| = m · 2^E`, `m` its integer
/// significand (the implicit bit included for a normal number).
fn decompose(x: f64) -> (u64, i64) {
    let bits = x.to_bits();
    let field = ((bits >> 52) & 0x7ff) as i64;
    let fraction = bits & ((1u64 << 52) - 1);
    if field == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), field - 1075)
    }
}

/// `x · 2^e` rounded to the nearest integer, ties to even, in exact integer
/// steps; `|x · 2^e| <= 2^43` by the scale rule.
fn quantize_value(x: f64, e: i64) -> i64 {
    let (m, exponent) = decompose(x);
    if m == 0 {
        return 0;
    }
    let shift = exponent + e;
    let magnitude = if shift >= 0 {
        // The scale rule bounds the result by 2^43, so the shift is small.
        m << shift
    } else if shift < -64 {
        0
    } else {
        let down = (-shift) as u32;
        let q = if down == 64 { 0 } else { m >> down };
        let remainder = if down == 64 {
            m
        } else {
            m & ((1u64 << down) - 1)
        };
        let half = 1u64 << (down - 1);
        match remainder.cmp(&half) {
            Ordering::Greater => q + 1,
            Ordering::Equal => q + (q & 1),
            Ordering::Less => q,
        }
    };
    let magnitude = magnitude as i64;
    if x.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

/// Quantize `column` into column `j` of the row-major `[n, k]` `out` and
/// return its exponent: `q = round_half_even(x · 2^e)` with `e` the largest
/// integer keeping `max|x| · 2^e <= 2^43` (ADR 0013).
///
/// # Errors
///
/// [`Error::NonFiniteValue`] (`field`) at the first value that is not
/// finite.
///
/// # Panics
///
/// If `out` does not have `column.len() · k` entries or `j >= k`; the host
/// sizes it.
pub fn quantize_column(
    column: &[f64],
    out: &mut [i64],
    k: usize,
    j: usize,
    field: &str,
) -> Result<i64, Error> {
    assert!(
        j < k && out.len() == column.len() * k,
        "quantize_column: out is [n, k]"
    );
    if let Some(position) = column.iter().position(|x| !x.is_finite()) {
        return Err(Error::NonFiniteValue {
            field: field.to_string(),
            position,
        });
    }
    let magnitude = column.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let e = exponent_of(magnitude);
    for (row, &x) in column.iter().enumerate() {
        out[row * k + j] = quantize_value(x, e);
    }
    Ok(e)
}

/// An unsigned magnitude [`round_ratio`] can divide: `u128` on the fast
/// path, [`BigUint`] when a value outgrows it.
trait Magnitude: Sized + Ord {
    fn bits(&self) -> u64;
    fn is_zero(&self) -> bool;
    /// `self · 2^s`, or `None` when it does not fit the type.
    fn shl(&self, s: u64) -> Option<Self>;
    fn div_rem(&self, divisor: &Self) -> (Self, Self);
    fn minus(&self, other: &Self) -> Self;
    fn to_u64(&self) -> Option<u64>;
}

impl Magnitude for u128 {
    fn bits(&self) -> u64 {
        u64::from(128 - self.leading_zeros())
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
    fn shl(&self, s: u64) -> Option<u128> {
        if *self == 0 {
            Some(0)
        } else if s < u64::from(self.leading_zeros()) {
            Some(self << s)
        } else {
            None
        }
    }
    fn div_rem(&self, divisor: &u128) -> (u128, u128) {
        (self / divisor, self % divisor)
    }
    fn minus(&self, other: &u128) -> u128 {
        self - other
    }
    fn to_u64(&self) -> Option<u64> {
        u64::try_from(*self).ok()
    }
}

impl Magnitude for BigUint {
    fn bits(&self) -> u64 {
        BigUint::bits(self)
    }
    fn is_zero(&self) -> bool {
        Zero::is_zero(self)
    }
    fn shl(&self, s: u64) -> Option<BigUint> {
        Some(self << s)
    }
    fn div_rem(&self, divisor: &BigUint) -> (BigUint, BigUint) {
        Integer::div_rem(self, divisor)
    }
    fn minus(&self, other: &BigUint) -> BigUint {
        self - other
    }
    fn to_u64(&self) -> Option<u64> {
        ToPrimitive::to_u64(self)
    }
}

/// `±a / (b · 2^shift)` rounded once to the nearest float64, ties to even;
/// the inner `None` when the rounded value is past the float64 range, the
/// outer `None` when an intermediate does not fit `M` (never for
/// [`BigUint`]).  `b` is positive.
fn round_ratio<M: Magnitude>(a: &M, b: &M, negative: bool, shift: i64) -> Option<Option<f64>> {
    let signed = |x: f64| if negative { -x } else { x };
    if a.is_zero() {
        return Some(Some(0.0));
    }
    let d = a.bits() as i64 - b.bits() as i64;
    // 2^(E-1) <= a / b < 2^E.
    let at_least = if d >= 0 {
        *a >= b.shl(d as u64)?
    } else {
        a.shl((-d) as u64)? >= *b
    };
    let e = d + i64::from(at_least) - shift;
    if e <= -1075 {
        return Some(Some(signed(0.0)));
    }
    if e > 1024 {
        return Some(None);
    }
    // The unit in the last place of the result: 53 significant bits, or
    // fewer below the normal range.
    let ulp = (e - 53).max(-1074);
    let t = ulp + shift;
    let (q, r, divisor) = if t >= 0 {
        let divisor = b.shl(t as u64)?;
        let (q, r) = a.div_rem(&divisor);
        (q, r, divisor)
    } else {
        let (q, r) = a.shl((-t) as u64)?.div_rem(b);
        (q, r, b.shl(0)?)
    };
    let mut q = q.to_u64()?;
    // r against divisor - r is 2r against the divisor, without overflow.
    match r.cmp(&divisor.minus(&r)) {
        Ordering::Greater => q += 1,
        Ordering::Equal => q += q & 1,
        Ordering::Less => {}
    }
    // q <= 2^53, and q >= 2^52 unless ulp is the subnormal unit, so the
    // biased exponent and significand add up in one integer; e <= 1024
    // keeps the shift in range.
    let bits = (((ulp + 1074) as u64) << 52) + q;
    if bits >= 0x7ff0_0000_0000_0000 {
        return Some(None);
    }
    Some(Some(signed(f64::from_bits(bits))))
}

/// `numerator / (denominator · 2^shift)` rounded once to the nearest
/// float64, ties to even, as Python's int true division rounds; `None` when
/// the rounded value is past the float64 range.  `denominator` is positive.
///
/// Subnormal results keep their reduced precision exactly, and a result
/// below half the smallest subnormal is a zero of the numerator's sign.
pub fn ratio(numerator: &BigInt, denominator: &BigUint, shift: i64) -> Option<f64> {
    debug_assert!(!Zero::is_zero(denominator));
    let negative = numerator.sign() == Sign::Minus;
    round_ratio(numerator.magnitude(), denominator, negative, shift).flatten()
}

/// [`ratio`] on `i128` operands; the outer `None` when an intermediate
/// outgrows `u128`.
fn ratio_i128(numerator: i128, denominator: u128, shift: i64) -> Option<Option<f64>> {
    round_ratio(
        &numerator.unsigned_abs(),
        &denominator,
        numerator < 0,
        shift,
    )
}

/// `n · ab − a · b`, or `None` when it outgrows `i128`.
fn centered_i128(n: i128, ab: i128, a: i128, b: i128) -> Option<i128> {
    n.checked_mul(ab)?.checked_sub(a.checked_mul(b)?)
}

/// `v · 2^s`, or `None` when it outgrows `i128`.
fn shl_i128(v: i128, s: u64) -> Option<i128> {
    if v == 0 {
        return Some(0);
    }
    let shifted = v.checked_shl(u32::try_from(s).ok()?)?;
    (shifted >> s == v).then_some(shifted)
}

/// Pearson's r from its exact numerators, `aa` and `bb` positive: the
/// ratio `ab² / (aa · bb)` is at most 1, so it never overflows.
fn pearson_of(ab: &BigInt, aa: &BigInt, bb: &BigInt) -> f64 {
    let squared = ratio(&(ab * ab), (aa * bb).magnitude(), 0).unwrap_or(f64::INFINITY);
    let r = squared.sqrt();
    if ab.is_negative() {
        -r
    } else {
        r
    }
}

/// Bytes the minimal two's complement form of `v` needs.
fn needed_i128(v: i128) -> usize {
    let magnitude = if v < 0 { !v } else { v };
    (128 - magnitude.leading_zeros() as usize + 1).div_ceil(8)
}

/// `values` at the smallest width that holds every one, as `(width,
/// bytes)`: the engine's accumulators as a host receives them.  The bytes
/// are allocated once, at that width, under `family`.
///
/// # Errors
///
/// [`Error::AllocationFailed`] for the bytes.
pub fn encode_i128(values: &[i128], family: Family) -> Result<(usize, Vec<u8>), Error> {
    let width = values.iter().map(|&v| needed_i128(v)).max().unwrap_or(1);
    let mut bytes = alloc::with_capacity(values.len() * width, family)?;
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes()[..width]);
    }
    Ok((width, bytes))
}

/// `values` at the smallest width that holds every one, as `(width,
/// bytes)`: a table a host builds from exact integers.
pub fn encode_big(values: &[BigInt]) -> (usize, Vec<u8>) {
    let encoded: Vec<Vec<u8>> = values.iter().map(BigInt::to_signed_bytes_le).collect();
    let width = encoded.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut bytes = Vec::with_capacity(values.len() * width);
    for (value, mut slot) in values.iter().zip(encoded) {
        slot.resize(width, if value.is_negative() { 0xff } else { 0x00 });
        bytes.extend_from_slice(&slot);
    }
    (width, bytes)
}

/// Write `v` into `slot`, which holds it.
fn write_i128(slot: &mut [u8], v: i128) {
    let bytes = v.to_le_bytes();
    if slot.len() <= 16 {
        slot.copy_from_slice(&bytes[..slot.len()]);
    } else {
        slot[..16].copy_from_slice(&bytes);
        slot[16..].fill(if v < 0 { 0xff } else { 0x00 });
    }
}

/// One float statistic a table offers per cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Statistic {
    /// `Σx` of a column over one member.
    Sum(Side),
    /// `Σx²` of a column over one member.
    Sumsq(Side),
    /// `Σ x_a y_b` of a product.
    Cross,
    /// `Σ(x − x̄)²` of a column over one member; `0` where a cell is empty.
    M2(Side),
    /// `Σ(x − x̄)(y − ȳ)` of a product; `0` where a cell is empty.
    Comoment,
    /// The mean of a column over one member; NaN where a cell is empty.
    Mean(Side),
    /// Pearson's r of a product; NaN where a cell is empty or an operand
    /// is constant.
    Pearson,
}

impl Statistic {
    pub const ALL: [Statistic; 11] = [
        Statistic::Sum(Side::First),
        Statistic::Sum(Side::Second),
        Statistic::Sumsq(Side::First),
        Statistic::Sumsq(Side::Second),
        Statistic::Cross,
        Statistic::M2(Side::First),
        Statistic::M2(Side::Second),
        Statistic::Comoment,
        Statistic::Mean(Side::First),
        Statistic::Mean(Side::Second),
        Statistic::Pearson,
    ];

    /// The host spelling: `"sum_first"`, `"cross"`, `"mean_second"`, ...
    pub fn name(self) -> &'static str {
        match self {
            Statistic::Sum(Side::First) => "sum_first",
            Statistic::Sum(Side::Second) => "sum_second",
            Statistic::Sumsq(Side::First) => "sumsq_first",
            Statistic::Sumsq(Side::Second) => "sumsq_second",
            Statistic::Cross => "cross",
            Statistic::M2(Side::First) => "m2_first",
            Statistic::M2(Side::Second) => "m2_second",
            Statistic::Comoment => "comoment",
            Statistic::Mean(Side::First) => "mean_first",
            Statistic::Mean(Side::Second) => "mean_second",
            Statistic::Pearson => "pearson",
        }
    }

    /// The statistic by its host spelling.
    pub fn parse(name: &str) -> Option<Statistic> {
        Statistic::ALL.into_iter().find(|s| s.name() == name)
    }

    /// Whether the statistic is indexed by product rather than by column.
    pub fn per_product(self) -> bool {
        matches!(
            self,
            Statistic::Cross | Statistic::Comoment | Statistic::Pearson
        )
    }

    /// The name an unrepresentable-output error gives the statistic; the
    /// mean names its operand (`"first.x"`) instead of its side.
    fn label(self) -> &'static str {
        match self {
            Statistic::Mean(_) => "mean",
            other => other.name(),
        }
    }
}

/// An exact relationship-moments table.
///
/// `shape` is the axis sizes, category axis included while it is present;
/// each cell holds `stride = 1 + 4k + p` accumulators: the pair count, the
/// sums of the first member's `k` columns, of the second's, their sums of
/// squares in the same order, and the cross sums of the `p` products, in
/// quantized units of `2^-exponents[c]` per column (`2^-(e_a + e_b)` for a
/// cross sum).  Cells are in row-major order over `shape`.  Each count is
/// in `[0, 2^63 − 1]`.  A table built from host bytes borrows them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MomentsTable<'a> {
    shape: Vec<usize>,
    layout: CellLayout,
    products: Vec<Product>,
    exponents: Vec<i64>,
    width: usize,
    bytes: Cow<'a, [u8]>,
}

fn invalid(reason: String) -> Error {
    Error::InvalidMomentsTable { reason }
}

fn count_overflow(operation: &'static str) -> Error {
    Error::ArithmeticOverflow {
        operation,
        dtype: "int64",
    }
}

/// Bytes the minimal two's complement form of `slot` needs.
fn needed(slot: &[u8]) -> usize {
    let top = slot[slot.len() - 1];
    let sign = if top & 0x80 == 0 { 0x00 } else { 0xff };
    let mut m = slot.len();
    while m > 1 && slot[m - 1] == sign && (slot[m - 2] & 0x80) == (sign & 0x80) {
        m -= 1;
    }
    m
}

fn write_slot(slot: &mut [u8], value: &BigInt) {
    let encoded = value.to_signed_bytes_le();
    debug_assert!(
        encoded.len() <= slot.len(),
        "an output width bound is wrong"
    );
    let fill = if value.sign() == Sign::Minus {
        0xff
    } else {
        0x00
    };
    slot[..encoded.len()].copy_from_slice(&encoded);
    slot[encoded.len()..].fill(fill);
}

/// Bytes that hold `bits` more bits.
fn bytes_for(bits: u64) -> usize {
    bits.div_ceil(8) as usize
}

impl<'a> MomentsTable<'a> {
    /// A table over accumulators of `width` bytes each, checked: the layout
    /// against the byte count, the products against the columns, every
    /// count in `[0, 2^63 − 1]`.  The bytes are kept as given, at any width.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidMomentsTable`] for an inconsistent table.
    pub fn from_bytes(
        shape: Vec<usize>,
        n_columns: usize,
        products: Vec<Product>,
        exponents: Vec<i64>,
        width: usize,
        bytes: Cow<'a, [u8]>,
    ) -> Result<MomentsTable<'a>, Error> {
        if width == 0 {
            return Err(invalid("width must be at least one byte".to_string()));
        }
        if exponents.len() != n_columns {
            return Err(invalid(format!(
                "{} exponents for {n_columns} columns",
                exponents.len()
            )));
        }
        if let Some(e) = exponents.iter().find(|e| e.abs() > MAX_EXPONENT) {
            return Err(invalid(format!(
                "exponent {e} is outside [-{MAX_EXPONENT}, {MAX_EXPONENT}]"
            )));
        }
        if let Some(column) = products
            .iter()
            .flat_map(|p| [p.a.column, p.b.column])
            .find(|&c| c >= n_columns)
        {
            return Err(invalid(format!(
                "a product names column {column} of {n_columns}"
            )));
        }
        let layout = CellLayout::new(n_columns, products.len()).ok_or_else(|| {
            invalid(format!(
                "{n_columns} columns and {} products overflow a cell",
                products.len()
            ))
        })?;
        let stride = layout.stride();
        let expected = shape
            .iter()
            .try_fold(stride, |acc, &s| acc.checked_mul(s))
            .and_then(|acc| acc.checked_mul(width));
        if expected != Some(bytes.len()) {
            return Err(invalid(format!(
                "{} bytes do not hold {:?} cells of {stride} accumulators of {width} bytes",
                bytes.len(),
                shape
            )));
        }
        let table = MomentsTable {
            shape,
            layout,
            products,
            exponents,
            width,
            bytes,
        };
        let max_count = BigInt::from(i64::MAX);
        for cell in 0..table.cells() {
            let valid = match table.read_i128(cell, 0) {
                Some(n) => (0..=i128::from(i64::MAX)).contains(&n),
                None => {
                    let n = table.read(cell, 0);
                    !n.is_negative() && n <= max_count
                }
            };
            if !valid {
                return Err(invalid(format!(
                    "cell {cell} has pair count {}, outside [0, 2^63 - 1]",
                    table.read(cell, 0)
                )));
            }
        }
        Ok(table)
    }

    /// A table of `values`, encoded at their minimal width.
    ///
    /// # Errors
    ///
    /// As [`MomentsTable::from_bytes`], and [`Error::AllocationFailed`].
    pub fn from_i128(
        shape: Vec<usize>,
        n_columns: usize,
        products: Vec<Product>,
        exponents: Vec<i64>,
        values: &[i128],
    ) -> Result<MomentsTable<'static>, Error> {
        let (width, bytes) = encode_i128(values, Family::MomentTable)?;
        MomentsTable::from_bytes(
            shape,
            n_columns,
            products,
            exponents,
            width,
            Cow::Owned(bytes),
        )
    }

    /// The axis sizes.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The value columns.
    pub fn n_columns(&self) -> usize {
        self.layout.n_columns()
    }

    /// The products, as operand pairs.
    pub fn products(&self) -> &[Product] {
        &self.products
    }

    /// The scale exponent of each column.
    pub fn exponents(&self) -> &[i64] {
        &self.exponents
    }

    /// Bytes per accumulator.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The accumulators, `width` bytes each, little-endian two's complement.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The accumulators, consuming the table.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes.into_owned()
    }

    /// Accumulators per cell.
    pub fn stride(&self) -> usize {
        self.layout.stride()
    }

    /// Cells: the product of the axis sizes.
    pub fn cells(&self) -> usize {
        self.shape.iter().product()
    }

    fn slot(&self, cell: usize, at: usize) -> &[u8] {
        let start = (cell * self.stride() + at) * self.width;
        &self.bytes[start..start + self.width]
    }

    /// Accumulator `at` of `cell`, decoded.
    pub fn read(&self, cell: usize, at: usize) -> BigInt {
        BigInt::from_signed_bytes_le(self.slot(cell, at))
    }

    /// Accumulator `at` of `cell` as an `i128`, which every accumulator of
    /// a table at most 16 bytes wide is; `None` for a wider table.
    #[inline]
    fn read_i128(&self, cell: usize, at: usize) -> Option<i128> {
        if self.width > 16 {
            return None;
        }
        let slot = self.slot(cell, at);
        let mut bytes = if slot[slot.len() - 1] & 0x80 == 0 {
            [0u8; 16]
        } else {
            [0xffu8; 16]
        };
        bytes[..slot.len()].copy_from_slice(slot);
        Some(i128::from_le_bytes(bytes))
    }

    /// The pair count of `cell`: non-negative and below 2^63, so its low
    /// eight bytes hold it.
    fn count(&self, cell: usize) -> i64 {
        let slot = self.slot(cell, 0);
        let mut bytes = [0u8; 8];
        let used = slot.len().min(8);
        bytes[..used].copy_from_slice(&slot[..used]);
        i64::from_le_bytes(bytes)
    }

    /// The pair count of every cell.
    pub fn counts(&self) -> Vec<i64> {
        (0..self.cells()).map(|cell| self.count(cell)).collect()
    }

    /// Every cell's pair count added exactly, which may pass 2^63.
    pub fn total_pairs(&self) -> u128 {
        (0..self.cells()).map(|cell| self.count(cell) as u128).sum()
    }

    /// Store an owned table at the smallest width that holds every
    /// accumulator, in place; a borrowed one stays as the host gave it.
    fn narrow(&mut self) {
        let Cow::Owned(bytes) = &mut self.bytes else {
            return;
        };
        let width = bytes
            .chunks_exact(self.width)
            .map(needed)
            .max()
            .unwrap_or(1);
        if width == self.width {
            return;
        }
        let accumulators = bytes.len() / self.width;
        for i in 0..accumulators {
            bytes.copy_within(i * self.width..i * self.width + width, i * width);
        }
        // Truncated, not shrunk: a shrinking realloc may copy, and every
        // host copies the output into its own buffer next anyway.
        bytes.truncate(accumulators * width);
        self.width = width;
    }

    fn with_bytes(
        &self,
        shape: Vec<usize>,
        exponents: Vec<i64>,
        width: usize,
    ) -> Result<MomentsTable<'static>, Error> {
        let cells: usize = shape.iter().product();
        let len = cells * self.stride() * width;
        Ok(MomentsTable {
            shape,
            layout: self.layout,
            products: self.products.clone(),
            exponents,
            width,
            bytes: Cow::Owned(alloc::filled(0u8, len, Family::MomentTable)?),
        })
    }

    fn check_axis(&self, axis: usize) -> Result<(), Error> {
        if axis >= self.shape.len() {
            return Err(invalid(format!(
                "axis {axis} of a table with {} axes",
                self.shape.len()
            )));
        }
        Ok(())
    }

    /// Keep the cells at `positions` of `axis`, in that order.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidMomentsTable`] for an axis or position out of range;
    /// [`Error::AllocationFailed`] for the output.
    pub fn select(&self, axis: usize, positions: &[usize]) -> Result<MomentsTable<'static>, Error> {
        self.check_axis(axis)?;
        let len = self.shape[axis];
        if let Some(p) = positions.iter().find(|&&p| p >= len) {
            return Err(invalid(format!(
                "position {p} of axis {axis} of size {len}"
            )));
        }
        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();
        let block = inner * self.stride() * self.width;
        let mut shape = self.shape.clone();
        shape[axis] = positions.len();
        let mut out = self.with_bytes(shape, self.exponents.clone(), self.width)?;
        let bytes = out.bytes.to_mut();
        for o in 0..outer {
            for (j, &p) in positions.iter().enumerate() {
                let from = (o * len + p) * block;
                let to = (o * positions.len() + j) * block;
                bytes[to..to + block].copy_from_slice(&self.bytes[from..from + block]);
            }
        }
        out.narrow();
        Ok(out)
    }

    /// Fold `axis` away by adding its cells exactly; an empty axis folds to
    /// zeros.
    ///
    /// # Errors
    ///
    /// [`Error::ArithmeticOverflow`] when a cell's pair count would pass
    /// `2^63 − 1` (nothing is returned); [`Error::InvalidMomentsTable`] for
    /// an axis out of range; [`Error::AllocationFailed`] for the output.
    pub fn sum(&self, axis: usize) -> Result<MomentsTable<'static>, Error> {
        self.check_axis(axis)?;
        let len = self.shape[axis];
        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();
        let stride = self.stride();
        let mut shape = self.shape.clone();
        shape.remove(axis);
        let width = self.width + bytes_for(u64::from(usize::BITS - len.leading_zeros()));
        let mut out = self.with_bytes(shape, self.exponents.clone(), width)?;
        let bytes = out.bytes.to_mut();
        let source = |o: usize, j: usize, i: usize| (o * len + j) * inner + i;
        for o in 0..outer {
            for i in 0..inner {
                for at in 0..stride {
                    let start = ((o * inner + i) * stride + at) * width;
                    let slot = &mut bytes[start..start + width];
                    let small = (0..len).try_fold(0i128, |total, j| {
                        total.checked_add(self.read_i128(source(o, j, i), at)?)
                    });
                    if let Some(total) = small {
                        if at == 0 && total > i128::from(i64::MAX) {
                            return Err(count_overflow("moments_sum"));
                        }
                        write_i128(slot, total);
                    } else {
                        let total: BigInt = (0..len).map(|j| self.read(source(o, j, i), at)).sum();
                        if at == 0 && total > BigInt::from(i64::MAX) {
                            return Err(count_overflow("moments_sum"));
                        }
                        write_slot(slot, &total);
                    }
                }
            }
        }
        out.narrow();
        Ok(out)
    }

    /// The shift each accumulator of a cell takes to move from `exponents`
    /// to `target`: none for the count, `s_c` for a sum, `2 s_c` for a sum of
    /// squares, `s_a + s_b` for a cross sum.
    fn shifts(&self, target: &[i64]) -> Vec<u64> {
        let s: Vec<u64> = target
            .iter()
            .zip(&self.exponents)
            .map(|(t, e)| (t - e) as u64)
            .collect();
        self.layout
            .slots()
            .map(|slot| match slot {
                Slot::Count => 0,
                Slot::Sum(c) => s[c],
                Slot::Square(c) => 2 * s[c],
                Slot::Cross(i) => {
                    let p = self.products[i];
                    s[p.a.column] + s[p.b.column]
                }
            })
            .collect()
    }

    /// Add two tables of one layout cell by cell.  Columns whose exponents
    /// differ are aligned first: the coarser operand's integers are shifted
    /// to the finer exponent, so the result carries the larger exponent of
    /// each column and no rounding.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidMomentsTable`] when the shapes, columns or products
    /// differ; [`Error::ArithmeticOverflow`] when a cell's pair count would
    /// pass `2^63 − 1`; [`Error::AllocationFailed`] for the output.
    pub fn merge(&self, other: &MomentsTable<'_>) -> Result<MomentsTable<'static>, Error> {
        if self.shape != other.shape
            || self.layout != other.layout
            || self.products != other.products
        {
            return Err(invalid(
                "merge needs two tables with the same shape, columns and products".to_string(),
            ));
        }
        let exponents: Vec<i64> = self
            .exponents
            .iter()
            .zip(&other.exponents)
            .map(|(a, b)| *a.max(b))
            .collect();
        let (shift_a, shift_b) = (self.shifts(&exponents), other.shifts(&exponents));
        let grown = |table: &MomentsTable<'_>, shifts: &[u64]| {
            table.width + bytes_for(shifts.iter().copied().max().unwrap_or(0))
        };
        let width = grown(self, &shift_a).max(grown(other, &shift_b)) + 1;
        let mut out = self.with_bytes(self.shape.clone(), exponents, width)?;
        let bytes = out.bytes.to_mut();
        let stride = self.stride();
        for cell in 0..self.cells() {
            for at in 0..stride {
                let start = (cell * stride + at) * width;
                let slot = &mut bytes[start..start + width];
                let small = (|| {
                    let a = shl_i128(self.read_i128(cell, at)?, shift_a[at])?;
                    a.checked_add(shl_i128(other.read_i128(cell, at)?, shift_b[at])?)
                })();
                if let Some(total) = small {
                    if at == 0 && total > i128::from(i64::MAX) {
                        return Err(count_overflow("moments_merge"));
                    }
                    write_i128(slot, total);
                } else {
                    let total = (self.read(cell, at) << shift_a[at])
                        + (other.read(cell, at) << shift_b[at]);
                    if at == 0 && total > BigInt::from(i64::MAX) {
                        return Err(count_overflow("moments_merge"));
                    }
                    write_slot(slot, &total);
                }
            }
        }
        out.narrow();
        Ok(out)
    }

    fn column_slots(&self, side: Side, column: usize) -> (usize, usize) {
        (
            self.layout.sum(side, column),
            self.layout.square(side, column),
        )
    }

    fn product_exponent(&self, index: usize) -> i64 {
        let product = self.products[index];
        self.exponents[product.a.column] + self.exponents[product.b.column]
    }

    /// The slots a product reads: its cross sum and each operand's sum.
    fn product_slots(&self, index: usize) -> (usize, usize, usize) {
        let product = self.products[index];
        let (sum_a, _) = self.column_slots(product.a.side, product.a.column);
        let (sum_b, _) = self.column_slots(product.b.side, product.b.column);
        (self.layout.cross(index), sum_a, sum_b)
    }

    /// One float per cell of `statistic` for column (or product) `index`.
    /// `name` is the column or product as the host spells it, for the
    /// error.
    ///
    /// # Errors
    ///
    /// [`Error::NotRepresentable`] when a value is past the float64 range;
    /// [`Error::InvalidMomentsTable`] for an index out of range;
    /// [`Error::AllocationFailed`] for the output.
    pub fn derive(
        &self,
        statistic: Statistic,
        index: usize,
        name: &str,
    ) -> Result<Vec<f64>, Error> {
        let limit = if statistic.per_product() {
            self.products.len()
        } else {
            self.layout.n_columns()
        };
        if index >= limit {
            return Err(invalid(format!(
                "{} index {index} of {limit}",
                statistic.name()
            )));
        }
        let mut out = alloc::with_capacity(self.cells(), Family::MomentOutput)?;
        for cell in 0..self.cells() {
            let value = self
                .derive_i128(statistic, index, cell)
                .unwrap_or_else(|| self.derive_big(statistic, index, cell));
            out.push(value.ok_or_else(|| Error::NotRepresentable {
                statistic: statistic.label(),
                name: name.to_string(),
            })?);
        }
        Ok(out)
    }

    /// The statistic of one cell on `i128` operands; the outer `None` when
    /// an operand or intermediate outgrows them, the inner when the value
    /// is past the float64 range.
    fn derive_i128(&self, statistic: Statistic, index: usize, cell: usize) -> Option<Option<f64>> {
        let n = self.read_i128(cell, 0)?;
        let read = |at: usize| self.read_i128(cell, at);
        let e = |i: usize| self.exponents[i];
        match statistic {
            Statistic::Sum(side) => {
                ratio_i128(read(self.column_slots(side, index).0)?, 1, e(index))
            }
            Statistic::Sumsq(side) => {
                ratio_i128(read(self.column_slots(side, index).1)?, 1, 2 * e(index))
            }
            Statistic::Cross => ratio_i128(
                read(self.layout.cross(index))?,
                1,
                self.product_exponent(index),
            ),
            Statistic::Mean(_) if n == 0 => Some(Some(f64::NAN)),
            Statistic::Mean(side) => {
                ratio_i128(read(self.column_slots(side, index).0)?, n as u128, e(index))
            }
            Statistic::M2(_) | Statistic::Comoment if n == 0 => Some(Some(0.0)),
            Statistic::M2(side) => {
                let (sum, sumsq) = self.column_slots(side, index);
                let (sum, sumsq) = (read(sum)?, read(sumsq)?);
                ratio_i128(centered_i128(n, sumsq, sum, sum)?, n as u128, 2 * e(index))
            }
            Statistic::Comoment => {
                let (cross, a, b) = self.product_slots(index);
                let numerator = centered_i128(n, read(cross)?, read(a)?, read(b)?)?;
                ratio_i128(numerator, n as u128, self.product_exponent(index))
            }
            Statistic::Pearson => {
                let product = self.products[index];
                let own = |operand: Operand| {
                    let (sum, sumsq) = self.column_slots(operand.side, operand.column);
                    let sum = read(sum)?;
                    centered_i128(n, read(sumsq)?, sum, sum)
                };
                let (aa, bb) = (own(product.a)?, own(product.b)?);
                if aa <= 0 || bb <= 0 {
                    return Some(Some(f64::NAN));
                }
                let (cross, a, b) = self.product_slots(index);
                let ab = centered_i128(n, read(cross)?, read(a)?, read(b)?)?;
                Some(Some(pearson_of(
                    &BigInt::from(ab),
                    &BigInt::from(aa),
                    &BigInt::from(bb),
                )))
            }
        }
    }

    /// [`MomentsTable::derive_i128`] on big integers, for any table.
    fn derive_big(&self, statistic: Statistic, index: usize, cell: usize) -> Option<f64> {
        let one = BigUint::one();
        let n = self.read(cell, 0);
        let read = |at: usize| self.read(cell, at);
        let e = |i: usize| self.exponents[i];
        let centered = |n: &BigInt, ab: &BigInt, a: &BigInt, b: &BigInt| n * ab - a * b;
        match statistic {
            Statistic::Sum(side) => ratio(&read(self.column_slots(side, index).0), &one, e(index)),
            Statistic::Sumsq(side) => {
                ratio(&read(self.column_slots(side, index).1), &one, 2 * e(index))
            }
            Statistic::Cross => ratio(
                &read(self.layout.cross(index)),
                &one,
                self.product_exponent(index),
            ),
            Statistic::Mean(_) if n.is_zero() => Some(f64::NAN),
            Statistic::Mean(side) => ratio(
                &read(self.column_slots(side, index).0),
                n.magnitude(),
                e(index),
            ),
            Statistic::M2(_) | Statistic::Comoment if n.is_zero() => Some(0.0),
            Statistic::M2(side) => {
                let (sum, sumsq) = self.column_slots(side, index);
                let sum = read(sum);
                ratio(
                    &centered(&n, &read(sumsq), &sum, &sum),
                    n.magnitude(),
                    2 * e(index),
                )
            }
            Statistic::Comoment => {
                let (cross, a, b) = self.product_slots(index);
                let numerator = centered(&n, &read(cross), &read(a), &read(b));
                ratio(&numerator, n.magnitude(), self.product_exponent(index))
            }
            Statistic::Pearson => {
                let product = self.products[index];
                let own = |operand: Operand| {
                    let (sum, sumsq) = self.column_slots(operand.side, operand.column);
                    let sum = read(sum);
                    centered(&n, &read(sumsq), &sum, &sum)
                };
                let (aa, bb) = (own(product.a), own(product.b));
                if !(aa.is_positive() && bb.is_positive()) {
                    return Some(f64::NAN);
                }
                let (cross, a, b) = self.product_slots(index);
                let ab = centered(&n, &read(cross), &read(a), &read(b));
                Some(pearson_of(&ab, &aa, &bb))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    // Reference values from CPython 3.14: `num / (den << shift)` (or
    // `(num << -shift) / den`), `None` where it raises OverflowError;
    // `43 - frexp(x)[1] + (frexp(x)[0] == 0.5)`; `np.rint(np.ldexp(x, e))`.
    const RATIO: &[(&str, &str, i64, Option<u64>)] = &[
        ("1", "3", 0, Some(0x3fd5555555555555)),
        ("-1", "3", 0, Some(0xbfd5555555555555)),
        ("2", "3", 0, Some(0x3fe5555555555555)),
        ("100000000000000000000", "7", 0, Some(0x43e8c821ae865725)),
        ("0", "5", 10, Some(0x0000000000000000)),
        ("1", "1", 1074, Some(0x0000000000000001)),
        ("1", "1", 1075, Some(0x0000000000000000)),
        ("3", "1", 1076, Some(0x0000000000000001)),
        ("1", "1", 1076, Some(0x0000000000000000)),
        ("-1", "1", 1080, Some(0x8000000000000000)),
        ("18014398509481985", "1", 1, Some(0x4340000000000000)),
        ("18014398509481987", "1", 1, Some(0x4340000000000001)),
        ("9007199254740993", "1", 0, Some(0x4340000000000000)),
        ("9007199254740995", "1", 0, Some(0x4340000000000002)),
        ("-9007199254740993", "1", 0, Some(0xc340000000000000)),
        ("179769313486231590772930519078902473361797697894230657273430081157732675805500963132708477322407536021120113879871393357658789768814416622492847430639474124377767893424865485276302219601246094119453082952085005768838150682342462881473913110540827237163350510684586298239947245938479716304835356329624224137216", "1", 0, None),
        ("179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497792", "1", 0, None),
        ("179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368", "1", 0, Some(0x7fefffffffffffff)),
        ("9007199254740991", "1", -971, Some(0x7fefffffffffffff)),
        ("9007199254740992", "1", -971, None),
        ("1", "3", -1023, Some(0x7fc5555555555555)),
        ("1", "3", -1025, Some(0x7fe5555555555555)),
        ("5", "1", 1076, Some(0x0000000000000001)),
        ("4503599627370497", "2", 1074, Some(0x0008000000000000)),
        ("7", "1", 1077, Some(0x0000000000000001)),
        ("1", "7", 1070, Some(0x0000000000000002)),
        ("-702767320667188464096392105081018594219622979", "1369926019498833445382347126422105567907", -646, Some(0xe97f4f92326270cd)),
        ("-83205709946157188380932826092322762648001208990845317216414610", "4234676704370533553849", -843, Some(0xfcfcdf00d762841c)),
        ("-24296510382039117969962992522699108388107803707535538081059116538821871876698", "48289277776", -12, Some(0xce531c4fb27ceb31)),
        ("-1479933140636312086", "14511472311392202665986204234287373336686", 801, Some(0x894ed29a9c04b017)),
        ("-5474356180670073267652396341766537678849641137934734318960622297944656", "4958206894218136078262519017507637177", 661, Some(0x9d7b37d9bce2d21b)),
        ("-450881678843625700959018928491113164413", "52621200624384", 798, Some(0x933c59c0938951bd)),
        ("63484745052090846246922916357766070504306737817785768326126523618", "368603521158282647577640007776304172", -651, Some(0x6eb1640d6b2080e1)),
        ("31814620172599766780164656298363074096197382499480", "298", 964, Some(0x0d72b34e188cfd15)),
        ("3192502771570766883305093909403764636759215060634508763589788", "65210159509901259369919532749", 816, Some(0x13834f685a883275)),
        ("-517995754418076691082", "72933513329419624253270818529086701354948", -1194, None),
        ("699852739570742319325547698207080380921090563361263072010550526", "49897", -279, Some(0x5d71e02ddbf44863)),
        ("20910245400387776909498157700548801679303207028", "1258000941000040829762941325772746118", -306, Some(0x552ef5e4d7213429)),
        ("96381083968055515008059769901370143219350106469093231277738641072609008916", "3498613078580747193976388819444", 820, Some(0x15b3c3d61579e0fa)),
        ("-134223797905332687138792050480328087824450541144113082", "403493254661461546", 1255, Some(0x8000000000000000)),
        ("9780880241375709572991", "102533135087520588020108491686071609620", -475, Some(0x5a4b7eb7323c7eb3)),
        ("-150199024250144221023102724330147304601341487", "2", -174, Some(0xd3eaf0cd0b88705e)),
        ("366343149327280772087064386975692652263906044397114985780617319418993", "13373651962", 1253, Some(0x0000000000008ba5)),
        ("-3623817145092265694308101491220723911341", "4351693840542478871034883310726332243790", 680, Some(0x956aa5c8637a295c)),
        ("-6516667562449954212474849625787040706027690405876143070929653708819826", "28933043820279174032846790482191882", 885, Some(0x8ff5b06a2b55e96b)),
        ("-1243576572604086219275290395626210259960984181309430605629924", "243239002206341376053290207474", 289, Some(0xb44021e6d235ed6c)),
        ("-75558045261611208961567318042059862876560882356277162267439509556", "46634", 1243, Some(0x80000000810f260d)),
        ("88724612072961486859639091295931092498444125602382268884305406933353061504", "97596757369341", -570, Some(0x70021a76d5a1d930)),
        ("-2078726861838888408287538748830337344193068845792530946593", "68813092074", -7, Some(0xca052a5954c90a8d)),
        ("-5033823729493618741590453513911047865169488509777123394313908837410126306614", "20267202770573474", -837, None),
        ("-102661450815490448797850705985720067318977286638065", "4305", -1171, None),
        ("36516739348836432985069174470720022180175768496831316565379847673", "8549652739343182840915291008975233719", -365, Some(0x5c7b9a00f09d192d)),
        ("-41971905772194969022768530878295993", "3132522344861649840018426374288086646", 332, Some(0xaacb70ceb9d046bd)),
        ("-33670465862878214725585478711466498071324439174507459576900457710457058810380", "3255308198946268219594185544801356616", -638, Some(0xf01e656573490dc1)),
        ("-234240807061072356567113393984721686015218651129", "12589295114", -946, None),
        ("1608225428920714258663151500736873616108302689", "19451246457121966199047535890065245774794", -864, Some(0x76f42f7d03fdcd42)),
        ("20011406696775294018583164659027092967311157761722", "1969026", -645, Some(0x712d2aaa8623b34e)),
        ("-4934205780249502904055057528906627809624590382924562197747600949761", "638513", 397, Some(0xb3c33c57ee1c63d1)),
        ("-249877766306613302782681930404182425", "6546949893006363992530560925201", -858, Some(0xf682a2e19c980e5a)),
        ("264184357588050429703281887937195007920294002519936024415938182759", "1964709252", -863, None),
        ("309823619328951565836003228123012529796023765420452478921394071964", "1180658732630946062739193492711", 138, Some(0x3ea9450d217de39f)),
        ("22402797155672253751865168984691991359929289062750965127928726630847952460501", "9183521363987118160396001", 446, Some(0x2eba14930a05a2d4)),
        ("804565494083445359424", "2988", 1283, Some(0x0000000000000000)),
        ("20707145149575725365972842512093354", "35821083526478819026188280", -400, Some(0x5ac13a55c56404ae)),
        ("4840265123042646590186698505287655210634378735554327", "8902373959923447010484889898", 795, Some(0x132cc8939c9ffbf7)),
        ("-124529025227540815861790361875690541534552560580330788737688445152008", "232810551106056624765113340", -823, Some(0xfc088fa6d5992d75)),
        ("14170753166737497552413230999358878893847772865268740056972252", "35645221724073823727175411245065", -1043, None),
        ("-68037423468533532908873782478850865810927384471", "333098549696170043017994647811432431190065", -959, Some(0xfcf8ef011b72f5b2)),
        ("10169331487267349571234448763159009643118871843", "778342", -931, None),
        ("-10908905278330061642996001552078", "36105777448386746910", 442, Some(0xa6b19632aaa7707b)),
        ("470119193783097719092507148096643141886880419950035867582846376362", "60475815179621488631195075371183", -488, Some(0x6577f458b3606e6c)),
        ("-5079281144724676235349190420977813717546060108913813988119446401005555524589", "3369410328499485272144250", -258, Some(0xdab01dcee61a76ec)),
        ("-569016225616892077880011163039021888313931", "327841253477093082213657256", -954, Some(0xfeb8aa3deb3ee8be)),
        ("2625696003730151198186926996580514", "51", 68, Some(0x42444e944ac88669)),
        ("7011934788735465178252123083458611944359998639556240263453126902241507911", "122739270528077554699020854391", -687, Some(0x73f47e713fb04f80)),
        ("-3300966061125027663873042", "116533682698952383530396393940175", -1112, None),
        ("-293419446709757424542012042035788170384081683605034469081409124756410", "482211075217821562780495041109941686730", -1246, None),
        ("8125448212908932080678589799538803237370447038926878521025291315468", "827756989122171872703387722914", 774, Some(0x173d8a24e20705ea)),
        ("-5208678491159845961197496417001421188640508452573995891337162519953203456075", "25539420993", -237, Some(0xdc4efc44a74e85a0)),
        ("-38482117776496103916410465492811471287550623532358758080007568270509", "3321476862868847786607381748138", -46, Some(0xca816eb4924a97ec)),
        ("-442235178153330213231173002208989417005521796732569191294882910064295808", "480009940655258745", 890, Some(0x93833cdbd6ec0b61)),
        ("36846775626288479024874739504620672010140015980656763484519", "145161592953829326351298", 349, Some(0x31787176447b4272)),
        ("1046142895115740784776876906290903334985985124891458", "1779658736801302389738", 1021, Some(0x064dad91a8a0f39f)),
        ("-358502147968781503486007690539897848216550262169220439170953865908576604757869", "13921956986422655941256183947736611200318", 1160, Some(0x8000004d7dba9aa1)),
        ("-17945573825604554173842174308689458729568864128774647051273999902660751503913", "180107685783023658026581459425", -1143, None),
        ("-15502906369198700479429426193287130971764196386016021714539362247900626915228", "1122027102361831931479505062", 694, Some(0x9ec2e8640edfe434)),
        ("1299937760207109796678074368", "1", -324, Some(0x59d0cd221cbe1456)),
        ("2039784943811773988864", "1", 540, Some(0x229ba4ecf7c9caba)),
        ("2110866247497129856", "1", 582, Some(0x1f5d4b4dab126da0)),
        ("1044553560039581285510673334272", "1", 380, Some(0x2e6a5e44e8433cfa)),
        ("60860689305949908", "1", 464, Some(0x266b070f2edbbada)),
        ("45756292271845616386048", "1", -563, Some(0x67d360e83ae408ca)),
        ("415035733917684467361431204921344", "1", -903, Some(0x7f24767cc89d0976)),
        ("1814166537586782275071903793152", "1", 94, Some(0x4056e5e363491ad0)),
        ("3733841201050318798848", "1", 416, Some(0x2a694d2e96b3bc44)),
        ("2528593053978459392", "1", -902, Some(0x7c218baefe25f3fe)),
        ("1164486872171776286130176", "1", -215, Some(0x525ed2deb6f18b82)),
        ("41770114312266795974656", "1", -654, Some(0x6d81b0b990f6e994)),
        ("749820279040158656", "1", -469, Some(0x60f4cfcb24632558)),
        ("26167343263727369584640", "1", 10, Some(0x43f62a2359675592)),
        ("156306638448222372102144", "1", 727, Some(0x17508cb32e404c26)),
        ("63446578394221607376650240", "1", -355, Some(0x5b7a3daaf40b1ea6)),
        ("22520990800942262", "1", 256, Some(0x335400ae322b852e)),
        ("114567425672501270808933919883264", "1", 626, Some(0x1f76982d4562b77c)),
        ("175902091648748316216262656", "1", -699, Some(0x71123016efdb4414)),
        ("1032080653584851424560921047990272", "1", -36, Some(0x49097158399e43e2)),
    ];
    const EXPONENT: &[(u64, i64)] = &[
        (0x3ff0000000000000, 43),
        (0x3fe8000000000000, 43),
        (0x42a0000000000000, 0),
        (0x0000000000000001, 1117),
        (0x7fefffffffffffff, -981),
        (0x4008000000000000, 41),
        (0x000012688b70e62b, 1072),
        (0x3fe0000000000000, 44),
        (0x0010000000000000, 1065),
        (0x40fe240c9fbe76c9, 26),
        (0x0000000000000003, 1115),
    ];
    const QUANTIZE: &[(u64, i64, i64)] = &[
        (0x3fe0000000000000, 0, 0),
        (0x3ff8000000000000, 0, 2),
        (0x4004000000000000, 0, 2),
        (0xc004000000000000, 0, -2),
        (0xbfe0000000000000, 0, 0),
        (0x3fdfffffffffffff, 0, 0),
        (0x7fefffffffffffff, -981, 8796093022208),
        (0x0000000000000001, 1116, 4398046511104),
        (0x81a56e1fc2f8f359, 40, 0),
        (0x400921fb54442d18, 41, 6908435304715),
        (0xc05edd2f1a9fbe77, 36, -8483831719920),
        (0xe85aa06b6f7d73d0, -604, -7319074889565),
        (0xb2fc80dc09e46bea, 250, -7834943256859),
        (0xc763dc86882dfe1c, -77, -5459467701120),
        (0xc81cac6b2999c538, -88, -7881714460273),
        (0xc2dc0b827063d3f1, -4, -7708939655413),
        (0xe295bb49ef51bcda, -512, -5973535872111),
        (0xc3c4c09e910bdba3, -19, -5704381645559),
        (0x619ef632794b42dd, -496, 8510689399505),
        (0x5f34345a1b758afa, -458, 5553770650979),
        (0xe5e8fc4360debb6b, -565, -6867935311791),
        (0xb994d5b254228111, 144, -5727013111968),
        (0xafb37a5e812e723b, 302, -5354073115549),
        (0x64404589c741b417, -539, 4472712581229),
        (0x4ae364286d02e1c3, -133, 5330223972536),
        (0x553d409bf6b1cee5, -298, 8040832937076),
        (0xa69386e250426964, 448, -5367510864026),
        (0x622fa3ff76826482, -505, 8697306521753),
        (0x5034e516dc0fbb6b, -218, 5743540896751),
        (0xada7486186f46b36, 335, -6399910329627),
        (0x479b32a522d64014, -80, 7476083209616),
    ];

    #[test]
    fn ratio_matches_python_true_division() {
        for &(num, den, shift, expected) in RATIO {
            let num = BigInt::from_str(num).unwrap();
            let den = BigUint::from_str(den).unwrap();
            let got = ratio(&num, &den, shift).map(f64::to_bits);
            assert_eq!(got, expected, "{num} / ({den} * 2^{shift})");
        }
    }

    #[test]
    fn ratio_agrees_with_float_division_on_small_operands() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..10_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let a = (state >> 11) as i64 - (1 << 52);
            let b = (state % (1 << 40)) + 1;
            let got = ratio(&BigInt::from(a), &BigUint::from(b), 0).unwrap();
            assert_eq!(got.to_bits(), (a as f64 / b as f64).to_bits(), "{a} / {b}");
        }
    }

    #[test]
    fn ratio_signs_an_underflow_by_its_numerator() {
        let tiny = ratio(&BigInt::from(-1), &BigUint::one(), 2000).unwrap();
        assert_eq!(tiny.to_bits(), (-0.0f64).to_bits());
        let zero = ratio(&BigInt::zero(), &BigUint::one(), -2000).unwrap();
        assert_eq!(zero.to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn exponent_matches_frexp() {
        for &(x, expected) in EXPONENT {
            assert_eq!(
                exponent_of(f64::from_bits(x)),
                expected,
                "{}",
                f64::from_bits(x)
            );
        }
        assert_eq!(exponent_of(0.0), 0);
    }

    #[test]
    fn quantize_matches_rint_of_ldexp() {
        for &(x, e, expected) in QUANTIZE {
            assert_eq!(
                quantize_value(f64::from_bits(x), e),
                expected,
                "{} * 2^{e}",
                f64::from_bits(x)
            );
        }
    }

    #[test]
    fn quantize_column_fills_one_column_and_refuses_non_finite() {
        let column = [1.0, -0.5, 0.25];
        let mut out = vec![7i64; 6];
        let e = quantize_column(&column, &mut out, 2, 1, "values['x']").unwrap();
        assert_eq!(e, 43);
        assert_eq!(out, vec![7, 1 << 43, 7, -(1 << 42), 7, 1 << 41]);
        let err = quantize_column(&[1.0, f64::NAN], &mut [0; 2], 1, 0, "values['x']").unwrap_err();
        assert_eq!(
            err.to_string(),
            "values['x'] is not finite at position 1; mask with a factor level instead"
        );
        assert_eq!(quantize_column(&[], &mut [], 1, 0, "v").unwrap(), 0);
    }

    #[test]
    fn pack_labels_is_mixed_radix_over_sorted_levels() {
        let a = [5i64, -1, 5, 9];
        let b = [0i64, 1, 1, 0];
        let packed = pack_labels(&[&a, &b], 4, "first label").unwrap();
        assert_eq!(packed.levels, vec![vec![-1, 5, 9], vec![0, 1]]);
        assert_eq!(packed.n_labels, 6);
        assert_eq!(packed.labels, vec![2, 1, 3, 4]);
        let none = pack_labels(&[], 3, "first label").unwrap();
        assert_eq!((none.labels, none.n_labels), (vec![0, 0, 0], 1));
        let empty = pack_labels(&[&[]], 0, "first label").unwrap();
        assert_eq!((empty.n_labels, empty.levels), (1, vec![vec![]]));
    }

    #[test]
    fn pack_labels_refuses_past_the_int32_range() {
        let n = 50_000;
        let ids: Vec<i64> = (0..n as i64).collect();
        let err = pack_labels(&[&ids, &ids], n, "first label").unwrap_err();
        match err {
            Error::ValueOutOfRange {
                field,
                maximum,
                value,
                ..
            } => {
                assert_eq!((field, maximum), ("first label", i64::from(i32::MAX)));
                assert_eq!(value, 50_000 * 50_000 - 1);
            }
            other => panic!("{other:?}"),
        }
        let short = pack_labels(&[&ids[..3]], n, "first label").unwrap_err();
        assert!(matches!(short, Error::LengthMismatch { .. }));
    }

    const PRODUCTS: [Product; 2] = [
        Product {
            a: Operand {
                side: Side::First,
                column: 0,
            },
            b: Operand {
                side: Side::Second,
                column: 0,
            },
        },
        Product {
            a: Operand {
                side: Side::First,
                column: 1,
            },
            b: Operand {
                side: Side::First,
                column: 0,
            },
        },
    ];

    /// A table of random accumulators over `shape`, 2 columns, `PRODUCTS`.
    fn random_values(shape: &[usize], seed: u64, bits: u32) -> Vec<i128> {
        let mut state = seed | 1;
        let stride = 1 + 4 * 2 + PRODUCTS.len();
        let cells: usize = shape.iter().product();
        (0..cells * stride)
            .map(|i| {
                let mut next = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state
                };
                let wide = (i128::from(next()) << 64) | i128::from(next());
                let magnitude = wide & ((1i128 << bits) - 1);
                if i % stride == 0 {
                    magnitude & 0xffff
                } else if state & 1 == 1 {
                    -magnitude
                } else {
                    magnitude
                }
            })
            .collect()
    }

    fn table(shape: &[usize], exponents: &[i64], values: &[i128]) -> MomentsTable<'static> {
        MomentsTable::from_i128(
            shape.to_vec(),
            2,
            PRODUCTS.to_vec(),
            exponents.to_vec(),
            values,
        )
        .unwrap()
    }

    fn values(t: &MomentsTable<'_>) -> Vec<BigInt> {
        (0..t.cells())
            .flat_map(|cell| (0..t.stride()).map(move |at| (cell, at)))
            .map(|(cell, at)| t.read(cell, at))
            .collect()
    }

    fn big(values: &[i128]) -> Vec<BigInt> {
        values.iter().map(|&v| BigInt::from(v)).collect()
    }

    #[test]
    fn a_table_is_stored_at_its_minimal_width() {
        let t = table(&[2, 3], &[0, 0], &random_values(&[2, 3], 7, 20));
        assert_eq!(t.width(), 3);
        let wide = table(&[1], &[0, 0], &{
            let mut v = vec![0i128; 11];
            v[3] = -(1i128 << 100);
            v
        });
        assert_eq!(wide.width(), 13);
        assert_eq!(wide.read(0, 3), BigInt::from(-(1i128 << 100)));
        let zeros = table(&[2], &[0, 0], &[0; 22]);
        assert_eq!((zeros.width(), zeros.bytes().len()), (1, 22));
        assert_eq!(needed(&[0x80, 0x00]), 2);
        assert_eq!(needed(&[0x7f, 0x00, 0x00]), 1);
        assert_eq!(needed(&[0x80, 0xff]), 1);
        assert_eq!(needed(&[0x7f, 0xff]), 2);
    }

    #[test]
    fn from_bytes_borrows_any_width_and_checks_the_layout() {
        let t = table(&[2, 3], &[1, -2], &random_values(&[2, 3], 9, 40));
        let mut wide = Vec::new();
        for chunk in t.bytes().chunks_exact(t.width()) {
            let fill = if chunk[chunk.len() - 1] & 0x80 == 0 {
                0
            } else {
                0xff
            };
            wide.extend_from_slice(chunk);
            wide.extend(std::iter::repeat_n(fill, 16 - chunk.len()));
        }
        let borrowed = MomentsTable::from_bytes(
            vec![2, 3],
            2,
            PRODUCTS.to_vec(),
            vec![1, -2],
            16,
            Cow::Borrowed(&wide),
        )
        .unwrap();
        assert!(matches!(borrowed.bytes, Cow::Borrowed(_)));
        assert_eq!(values(&borrowed), values(&t));
        assert_eq!(borrowed.sum(0).unwrap(), t.sum(0).unwrap());

        let bad = |shape: Vec<usize>, exponents: Vec<i64>, width: usize, bytes: Vec<u8>| {
            MomentsTable::from_bytes(
                shape,
                2,
                PRODUCTS.to_vec(),
                exponents,
                width,
                Cow::Owned(bytes),
            )
            .unwrap_err()
            .to_string()
        };
        assert!(bad(vec![2], vec![0, 0], 1, vec![0; 21]).contains("21 bytes"));
        assert!(bad(vec![1], vec![0], 1, vec![0; 11]).contains("1 exponents for 2 columns"));
        assert!(bad(vec![1], vec![0, 1 << 17], 1, vec![0; 11]).contains("outside"));
        assert!(bad(vec![1], vec![0, 0], 0, vec![]).contains("width"));
        let mut negative = vec![0u8; 11];
        negative[0] = 0xff;
        assert!(bad(vec![1], vec![0, 0], 1, negative).contains("pair count -1"));
        let mut past = vec![0u8; 11 * 9];
        past[7] = 0x80; // 2^63 at width 9
        assert!(bad(vec![1], vec![0, 0], 9, past).contains("pair count 9223372036854775808"));
    }

    #[test]
    fn sum_adds_exactly_along_one_axis() {
        let shape = [3, 4, 2];
        let raw = random_values(&shape, 11, 100);
        let t = table(&shape, &[3, 5], &raw);
        let stride = t.stride();
        for axis in 0..3 {
            let folded = t.sum(axis).unwrap();
            let mut kept = shape.to_vec();
            kept.remove(axis);
            assert_eq!(folded.shape(), kept.as_slice());
            assert_eq!(folded.exponents(), &[3, 5]);
            let mut expected = vec![BigInt::zero(); folded.cells() * stride];
            for (i, &v) in raw.iter().enumerate() {
                let (cell, at) = (i / stride, i % stride);
                let index = [cell / 8, cell / 2 % 4, cell % 2];
                let out = match axis {
                    0 => index[1] * 2 + index[2],
                    1 => index[0] * 2 + index[2],
                    _ => index[0] * 4 + index[1],
                };
                expected[out * stride + at] += v;
            }
            assert_eq!(values(&folded), expected);
        }
        let empty = t.select(1, &[]).unwrap().sum(1).unwrap();
        assert!(values(&empty).iter().all(Zero::is_zero));
        assert_eq!(empty.shape(), &[3, 2]);
    }

    #[test]
    fn select_keeps_positions_in_order() {
        let shape = [2, 3];
        let raw = random_values(&shape, 13, 60);
        let t = table(&shape, &[0, 0], &raw);
        let picked = t.select(1, &[2, 0]).unwrap();
        assert_eq!(picked.shape(), &[2, 2]);
        let stride = t.stride();
        let cell = |r: usize, c: usize| &raw[(r * 3 + c) * stride..(r * 3 + c + 1) * stride];
        let expected: Vec<i128> = [cell(0, 2), cell(0, 0), cell(1, 2), cell(1, 0)].concat();
        assert_eq!(values(&picked), big(&expected));
        assert!(t.select(1, &[3]).is_err());
        assert!(t.select(2, &[0]).is_err());
    }

    #[test]
    fn merge_aligns_exponents_without_rounding() {
        let shape = [2, 2];
        let (ra, rb) = (random_values(&shape, 17, 80), random_values(&shape, 19, 80));
        let a = table(&shape, &[10, -4], &ra);
        let b = table(&shape, &[-1990, 7], &rb);
        let merged = a.merge(&b).unwrap();
        assert_eq!(merged.exponents(), &[10, 7]);
        // Shifts per slot: count 0; sums s_c; sumsq 2 s_c; cross s_a + s_b.
        let (sa, sb) = ([0u32, 11], [2000u32, 0]);
        let shift = |s: [u32; 2], at: usize| -> u32 {
            match at {
                0 => 0,
                1 | 3 => s[0],
                2 | 4 => s[1],
                5 | 7 => 2 * s[0],
                6 | 8 => 2 * s[1],
                9 => s[0] + s[0],
                _ => s[1] + s[0],
            }
        };
        let stride = a.stride();
        let expected: Vec<BigInt> = ra
            .iter()
            .zip(&rb)
            .enumerate()
            .map(|(i, (&x, &y))| {
                (BigInt::from(x) << shift(sa, i % stride))
                    + (BigInt::from(y) << shift(sb, i % stride))
            })
            .collect();
        assert_eq!(values(&merged), expected);
        assert!(merged.width() > 500);
        // The merge commutes, and folding then merging equals merging then folding.
        assert_eq!(b.merge(&a).unwrap(), merged);
        assert_eq!(
            a.sum(0).unwrap().merge(&b.sum(0).unwrap()).unwrap(),
            merged.sum(0).unwrap()
        );
        let other = table(&[4], &[0, 0], &random_values(&[4], 1, 8));
        assert!(a
            .merge(&other)
            .unwrap_err()
            .to_string()
            .contains("same shape"));
    }

    fn counts_table(counts: &[i64]) -> MomentsTable<'static> {
        let mut bytes = Vec::new();
        for &n in counts {
            bytes.extend_from_slice(&n.to_le_bytes());
            bytes.extend(std::iter::repeat_n(0u8, 8 * 10));
        }
        MomentsTable::from_bytes(
            vec![counts.len()],
            2,
            PRODUCTS.to_vec(),
            vec![0, 0],
            8,
            Cow::Owned(bytes),
        )
        .unwrap()
    }

    #[test]
    fn counts_are_checked_per_cell_and_totals_are_exact() {
        let max = counts_table(&[i64::MAX]);
        assert_eq!(max.counts(), vec![i64::MAX]);
        let overflow = |e: Error, op: &str| matches!(e, Error::ArithmeticOverflow { operation, dtype: "int64" } if operation == op);
        assert!(overflow(
            max.merge(&counts_table(&[1])).unwrap_err(),
            "moments_merge"
        ));
        assert!(overflow(
            counts_table(&[i64::MAX, 1]).sum(0).unwrap_err(),
            "moments_sum"
        ));
        let three = counts_table(&[i64::MAX, i64::MAX, 5]);
        assert_eq!(three.total_pairs(), 2 * i64::MAX as u128 + 5);
        assert_eq!(
            counts_table(&[i64::MAX - 1, 1]).sum(0).unwrap().counts(),
            vec![i64::MAX]
        );
    }

    /// Two columns per member with values `(x, y)` per pair, one cell.
    fn one_cell(pairs: &[([f64; 2], [f64; 2])]) -> MomentsTable<'static> {
        let mut first = vec![0i64; pairs.len() * 2];
        let mut second = vec![0i64; pairs.len() * 2];
        let mut exponents = vec![0i64; 2];
        for j in 0..2 {
            let both: Vec<f64> = pairs
                .iter()
                .map(|p| p.0[j])
                .chain(pairs.iter().map(|p| p.1[j]))
                .collect();
            let mut q = vec![0i64; both.len()];
            exponents[j] = quantize_column(&both, &mut q, 1, 0, "v").unwrap();
            for (i, &v) in q.iter().enumerate() {
                if i < pairs.len() {
                    first[i * 2 + j] = v;
                } else {
                    second[(i - pairs.len()) * 2 + j] = v;
                }
            }
        }
        let mut acc = vec![0i128; 11];
        for i in 0..pairs.len() {
            let (a, b) = (&first[i * 2..i * 2 + 2], &second[i * 2..i * 2 + 2]);
            acc[0] += 1;
            for c in 0..2 {
                let (x, y) = (i128::from(a[c]), i128::from(b[c]));
                acc[1 + c] += x;
                acc[3 + c] += y;
                acc[5 + c] += x * x;
                acc[7 + c] += y * y;
            }
            acc[9] += i128::from(a[0]) * i128::from(b[0]);
            acc[10] += i128::from(a[1]) * i128::from(a[0]);
        }
        table(&[1], &exponents, &acc)
    }

    #[test]
    fn derive_gives_exact_moments_of_representable_values() {
        let t = one_cell(&[
            ([1.0, 2.0], [3.0, 0.5]),
            ([2.0, 4.0], [5.0, 0.5]),
            ([6.0, 6.0], [1.0, 0.5]),
        ]);
        let get = |s: &str, i: usize| t.derive(Statistic::parse(s).unwrap(), i, "x").unwrap()[0];
        assert_eq!(get("sum_first", 0), 9.0);
        assert_eq!(get("sum_second", 1), 1.5);
        assert_eq!(get("sumsq_first", 1), 56.0);
        assert_eq!(get("mean_first", 0), 3.0);
        assert_eq!(get("m2_first", 0), 14.0);
        assert_eq!(get("m2_second", 1), 0.0);
        assert_eq!(get("cross", 0), 19.0);
        assert_eq!(get("cross", 1), 46.0);
        assert_eq!(get("comoment", 0), -8.0);
        assert_eq!(get("comoment", 1), 10.0);
        // N_ab = n·Σab − Σa·Σb: first.x × second.x is −24 over N_aa 42 and
        // N_bb 24; first.y × first.x is 30 over 24 and 42.
        assert_eq!(get("pearson", 0), -(576.0f64 / 1008.0).sqrt());
        assert_eq!(get("pearson", 1), (900.0f64 / 1008.0).sqrt());
        assert!(t.derive(Statistic::Cross, 2, "x").is_err());
        assert!(t.derive(Statistic::Sum(Side::First), 2, "x").is_err());
    }

    #[test]
    fn derive_marks_empty_and_constant_cells() {
        let t = one_cell(&[([1.0, 2.0], [1.0, 2.0])])
            .select(0, &[0])
            .unwrap();
        let empty = table(&[1], t.exponents(), &[0; 11]);
        assert!(empty.derive(Statistic::Mean(Side::First), 0, "x").unwrap()[0].is_nan());
        assert_eq!(
            empty.derive(Statistic::M2(Side::First), 0, "x").unwrap()[0],
            0.0
        );
        assert_eq!(empty.derive(Statistic::Comoment, 0, "x").unwrap()[0], 0.0);
        assert!(empty.derive(Statistic::Pearson, 0, "x").unwrap()[0].is_nan());
        assert!(t.derive(Statistic::Pearson, 0, "x").unwrap()[0].is_nan());
    }

    #[test]
    fn an_unrepresentable_output_names_the_statistic_and_column() {
        let (x, y) = (2f64.powi(660), 2f64.powi(661));
        let t = one_cell(&[([x, 1.0], [y, 1.0]), ([x, 1.0], [x, 1.0])]);
        let sum = t.derive(Statistic::Sum(Side::First), 0, "big").unwrap();
        assert_eq!(sum[0], y);
        let mean = t
            .derive(Statistic::Mean(Side::Second), 0, "second.big")
            .unwrap();
        assert_eq!(mean[0], 1.5 * x);
        let err = t
            .derive(Statistic::Sumsq(Side::First), 0, "big")
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "sumsq_first of 'big' is not representable in float64 (an output overflowed)"
        );
        assert_eq!(err.class(), crate::error::ErrorClass::Usage);
        let err = t.derive(Statistic::M2(Side::Second), 0, "big").unwrap_err();
        assert!(err.to_string().starts_with("m2_second of 'big'"));
        assert_eq!(
            t.derive(Statistic::M2(Side::First), 0, "big").unwrap()[0],
            0.0
        );
        let r = t.derive(Statistic::Pearson, 0, "x").unwrap()[0];
        assert!(r.is_nan(), "first.big is constant");
    }

    /// `t` with every accumulator padded to 20 bytes, so no read fits `i128`.
    fn padded(t: &MomentsTable<'_>) -> MomentsTable<'static> {
        let mut bytes = Vec::new();
        for slot in t.bytes().chunks_exact(t.width()) {
            let fill = if slot[slot.len() - 1] & 0x80 == 0 {
                0
            } else {
                0xff
            };
            bytes.extend_from_slice(slot);
            bytes.extend(std::iter::repeat_n(fill, 20 - slot.len()));
        }
        MomentsTable::from_bytes(
            t.shape().to_vec(),
            t.n_columns(),
            t.products().to_vec(),
            t.exponents().to_vec(),
            20,
            Cow::Owned(bytes),
        )
        .unwrap()
    }

    #[test]
    fn the_i128_and_big_integer_paths_agree() {
        // Up to 126 bits: some numerators and sums outgrow i128 and fall back.
        for (seed, bits) in [(3, 30), (5, 60), (7, 90), (11, 110), (13, 126)] {
            let shape = [3, 2, 2];
            let t = table(&shape, &[17, -40], &random_values(&shape, seed, bits));
            let wide = padded(&t);
            assert!(wide.read_i128(0, 0).is_none());
            for statistic in Statistic::ALL {
                for index in 0..2 {
                    let fast = t
                        .derive(statistic, index, "x")
                        .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
                    let big = wide
                        .derive(statistic, index, "x")
                        .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
                    assert_eq!(
                        fast.is_ok(),
                        big.is_ok(),
                        "{} {index} at {bits} bits",
                        statistic.name()
                    );
                    if let (Ok(fast), Ok(big)) = (fast, big) {
                        assert_eq!(fast, big, "{} {index} at {bits} bits", statistic.name());
                    }
                }
            }
            for axis in 0..3 {
                assert_eq!(
                    t.sum(axis).map(|x| values(&x)).ok(),
                    wide.sum(axis).map(|x| values(&x)).ok()
                );
            }
            let other = table(&shape, &[20, -45], &random_values(&shape, seed + 1, bits));
            assert_eq!(
                t.merge(&other).map(|x| values(&x)).ok(),
                wide.merge(&padded(&other)).map(|x| values(&x)).ok()
            );
            assert_eq!(t.merge(&other).unwrap(), wide.merge(&other).unwrap());
        }
    }

    #[test]
    fn statistic_names_round_trip() {
        for s in Statistic::ALL {
            assert_eq!(Statistic::parse(s.name()), Some(s));
        }
        assert_eq!(Statistic::parse("variance"), None);
    }
}
