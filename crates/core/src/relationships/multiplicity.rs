//! Path multiplicity saturated at two.
//!
//! The relationship engine only ever asks three questions of a path count:
//! is it zero, is it exactly one, is it at least two.  The map
//! `s(n) = min(n, 2)` from the natural numbers onto `{0, 1, 2}` respects both
//! semiring operations under saturation:
//!
//! * `s(a + b) = min(s(a) + s(b), 2)`: if either operand is at least two both
//!   sides are two, otherwise both sides are the exact sum.
//! * `s(a * b) = min(s(a) * s(b), 2)`: zero if either operand is zero;
//!   otherwise `a * b >= 2` exactly when `a >= 2` or `b >= 2`, which is exactly
//!   when `s(a) * s(b) >= 2`.
//!
//! So every sparse product in the engine can be evaluated in this saturated
//! arithmetic and still answer the three questions exactly.  Nothing can
//! overflow, which is the resolution of issue #9 for this engine.
//! [`multiplicity_is_a_homomorphism`] checks the identities exhaustively.

/// A path count that distinguishes only zero, one, and "at least two".
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Mult(u8);

impl Mult {
    pub const ZERO: Mult = Mult(0);
    pub const ONE: Mult = Mult(1);

    pub fn from_count(n: u64) -> Mult {
        Mult(n.min(2) as u8)
    }

    #[inline]
    pub fn is_one(self) -> bool {
        self.0 == 1
    }

    #[inline]
    pub fn at_least_two(self) -> bool {
        self.0 >= 2
    }
}

impl std::ops::Add for Mult {
    type Output = Mult;

    #[inline]
    fn add(self, other: Mult) -> Mult {
        Mult((self.0 + other.0).min(2))
    }
}

impl std::ops::Mul for Mult {
    type Output = Mult;

    #[inline]
    fn mul(self, other: Mult) -> Mult {
        Mult((self.0 * other.0).min(2))
    }
}

#[cfg(test)]
mod tests {
    use super::Mult;

    #[test]
    fn multiplicity_is_a_homomorphism() {
        for a in 0..8u64 {
            for b in 0..8u64 {
                let (sa, sb) = (Mult::from_count(a), Mult::from_count(b));
                assert_eq!(sa + sb, Mult::from_count(a + b), "add {a} {b}");
                assert_eq!(sa * sb, Mult::from_count(a * b), "mul {a} {b}");
            }
        }
    }
}
