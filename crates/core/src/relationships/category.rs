//! The 23 relationship categories in registry order (degree ascending, then precedence).

/// Relationship category codes, in the order of the Python `REL_REGISTRY`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Category {
    MZ,
    MO,
    FO,
    FS,
    MHS,
    PHS,
    GP,
    Av,
    GGP,
    HAv,
    GAv,
    C1,
    GGGP,
    HGAv,
    GGAv,
    H1C,
    C1R1,
    G3GP,
    HGGAv,
    G3Av,
    H1C1R,
    C1R2,
    C2,
}

pub const N_CATEGORIES: usize = 23;

impl Category {
    pub const ALL: [Category; N_CATEGORIES] = [
        Category::MZ,
        Category::MO,
        Category::FO,
        Category::FS,
        Category::MHS,
        Category::PHS,
        Category::GP,
        Category::Av,
        Category::GGP,
        Category::HAv,
        Category::GAv,
        Category::C1,
        Category::GGGP,
        Category::HGAv,
        Category::GGAv,
        Category::H1C,
        Category::C1R1,
        Category::G3GP,
        Category::HGGAv,
        Category::G3Av,
        Category::H1C1R,
        Category::C1R2,
        Category::C2,
    ];

    /// The short code used by the Python registry.
    pub fn code(self) -> &'static str {
        match self {
            Category::MZ => "MZ",
            Category::MO => "MO",
            Category::FO => "FO",
            Category::FS => "FS",
            Category::MHS => "MHS",
            Category::PHS => "PHS",
            Category::GP => "GP",
            Category::Av => "Av",
            Category::GGP => "GGP",
            Category::HAv => "HAv",
            Category::GAv => "GAv",
            Category::C1 => "1C",
            Category::GGGP => "GGGP",
            Category::HGAv => "HGAv",
            Category::GGAv => "GGAv",
            Category::H1C => "H1C",
            Category::C1R1 => "1C1R",
            Category::G3GP => "G3GP",
            Category::HGGAv => "HGGAv",
            Category::G3Av => "G3Av",
            Category::H1C1R => "H1C1R",
            Category::C1R2 => "1C2R",
            Category::C2 => "2C",
        }
    }

    /// The category by its registry code.
    pub fn parse(code: &str) -> Option<Category> {
        Category::ALL.into_iter().find(|cat| cat.code() == code)
    }

    /// Kinship degree: 0 for MZ, 1 for parent-offspring and full sibs, and so on.
    pub const fn degree(self) -> u8 {
        match self {
            Category::MZ => 0,
            Category::MO | Category::FO | Category::FS => 1,
            Category::MHS | Category::PHS | Category::GP | Category::Av => 2,
            Category::GGP | Category::HAv | Category::GAv | Category::C1 => 3,
            Category::GGGP | Category::HGAv | Category::GGAv | Category::H1C | Category::C1R1 => 4,
            Category::G3GP
            | Category::HGGAv
            | Category::G3Av
            | Category::H1C1R
            | Category::C1R2
            | Category::C2 => 5,
        }
    }

    /// Kinship coefficient the category's path shape implies, with no
    /// inbreeding or duplicate paths: `2^-(degree + 1)` for every category,
    /// which is `ancestor_count() * 2^-(up() + down() + 1)` for all but MZ.
    pub fn nominal_kinship(self) -> f64 {
        0.5f64.powi(i32::from(self.degree()) + 1)
    }

    /// `(up, down, ancestor_count)`: the category's path shape, one line per
    /// category.  [`up`](Self::up), [`down`](Self::down),
    /// [`ancestor_count`](Self::ancestor_count) and [`symmetric`](Self::symmetric)
    /// read it.
    pub const fn shape(self) -> (u8, u8, u8) {
        match self {
            Category::MZ => (0, 0, 0),
            Category::MO => (1, 0, 1),
            Category::FO => (1, 0, 1),
            Category::FS => (1, 1, 2),
            Category::MHS => (1, 1, 1),
            Category::PHS => (1, 1, 1),
            Category::GP => (2, 0, 1),
            Category::Av => (2, 1, 2),
            Category::GGP => (3, 0, 1),
            Category::HAv => (2, 1, 1),
            Category::GAv => (3, 1, 2),
            Category::C1 => (2, 2, 2),
            Category::GGGP => (4, 0, 1),
            Category::HGAv => (3, 1, 1),
            Category::GGAv => (4, 1, 2),
            Category::H1C => (2, 2, 1),
            Category::C1R1 => (3, 2, 2),
            Category::G3GP => (5, 0, 1),
            Category::HGGAv => (4, 1, 1),
            Category::G3Av => (5, 1, 2),
            Category::H1C1R => (3, 2, 1),
            Category::C1R2 => (4, 2, 2),
            Category::C2 => (3, 3, 2),
        }
    }

    /// Meioses from the `first` pair member up to the shared ancestor(s).
    ///
    /// `first` is the member at least as far from the ancestor(s), so
    /// `up() >= down()`, with equality exactly for a symmetric category.
    pub const fn up(self) -> u8 {
        self.shape().0
    }

    /// Meioses from the shared ancestor(s) down to the `second` pair member.
    pub const fn down(self) -> u8 {
        self.shape().1
    }

    /// The paths of this shape that join a pair: 2 through a mated pair of
    /// ancestors (full), 1 through a single ancestor (half and lineal), and 0
    /// for MZ, whose twins are not joined by a meiotic path.
    ///
    /// Apart from MZ, `nominal_kinship() == ancestor_count() * 2^-(up() +
    /// down() + 1)`.  The engine decides full against half by this count
    /// (ADR 0018).
    pub const fn ancestor_count(self) -> u8 {
        self.shape().2
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Whether the category has no roles, so a pair is stored `first < second`:
    /// both members are as far from the shared ancestor(s), `up() == down()`.
    /// Every other category names a `first_role` and a `second_role`.
    pub const fn symmetric(self) -> bool {
        let (up, down, _) = self.shape();
        up == down
    }

    /// The `(first, second)` semantic roles of an asymmetric category, as the
    /// Python registry names them; `None` for a symmetric one.
    pub fn roles(self) -> Option<(&'static str, &'static str)> {
        match self {
            Category::MO => Some(("offspring", "mother")),
            Category::FO => Some(("offspring", "father")),
            Category::GP | Category::GGP | Category::GGGP | Category::G3GP => {
                Some(("descendant", "ancestor"))
            }
            Category::Av
            | Category::HAv
            | Category::GAv
            | Category::HGAv
            | Category::GGAv
            | Category::HGGAv
            | Category::G3Av => Some(("niece_nephew", "aunt_uncle")),
            Category::C1R1 | Category::H1C1R | Category::C1R2 => {
                Some(("junior_cousin", "senior_cousin"))
            }
            Category::MZ
            | Category::FS
            | Category::MHS
            | Category::PHS
            | Category::C1
            | Category::H1C
            | Category::C2 => None,
        }
    }
}

/// A set of categories, for the requested blocks of a pair query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CategorySet([bool; N_CATEGORIES]);

impl CategorySet {
    pub const EMPTY: CategorySet = CategorySet([false; N_CATEGORIES]);

    /// Every category up to and including `degree`.
    pub fn up_to_degree(degree: u8) -> CategorySet {
        let mut set = CategorySet::EMPTY;
        for cat in Category::ALL {
            if cat.degree() <= degree {
                set.0[cat.index()] = true;
            }
        }
        set
    }

    pub fn insert(&mut self, cat: Category) {
        self.0[cat.index()] = true;
    }

    #[inline]
    pub fn contains(&self, cat: Category) -> bool {
        self.0[cat.index()]
    }

    /// The members in registry order.
    pub fn iter(&self) -> impl Iterator<Item = Category> + '_ {
        Category::ALL.into_iter().filter(|cat| self.contains(*cat))
    }

    /// The highest degree of any member, or `None` when empty.
    pub fn top_degree(&self) -> Option<u8> {
        self.iter().map(Category::degree).max()
    }
}

impl FromIterator<Category> for CategorySet {
    fn from_iter<I: IntoIterator<Item = Category>>(iter: I) -> CategorySet {
        let mut set = CategorySet::EMPTY;
        for cat in iter {
            set.insert(cat);
        }
        set
    }
}

/// Per-category pair counts, indexed by [`Category::index`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts(pub [u64; N_CATEGORIES]);

impl Counts {
    #[inline]
    pub fn add(&mut self, c: Category, n: u64) {
        self.0[c.index()] += n;
    }

    pub fn get(&self, c: Category) -> u64 {
        self.0[c.index()]
    }

    pub fn merge(mut self, other: Counts) -> Counts {
        for (a, b) in self.0.iter_mut().zip(other.0) {
            *a += b;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::Category;

    #[test]
    fn roles_exist_exactly_for_asymmetric_categories() {
        for cat in Category::ALL {
            assert_eq!(cat.roles().is_none(), cat.symmetric(), "{}", cat.code());
        }
    }

    #[test]
    fn first_is_at_least_as_far_from_the_ancestors() {
        for cat in Category::ALL {
            assert!(cat.up() >= cat.down(), "{}", cat.code());
            assert_eq!(cat.up() == cat.down(), cat.symmetric(), "{}", cat.code());
        }
    }

    #[test]
    fn nominal_kinship_follows_the_path_shape() {
        for cat in Category::ALL.into_iter().filter(|&c| c != Category::MZ) {
            let paths = f64::from(cat.ancestor_count());
            let meioses = i32::from(cat.up() + cat.down());
            let want = paths * 0.5f64.powi(meioses + 1);
            assert!(
                (cat.nominal_kinship() - want).abs() < 1e-15,
                "{}",
                cat.code()
            );
        }
    }

    #[test]
    fn mz_is_the_stated_exception() {
        let mz = Category::MZ;
        assert_eq!((mz.up(), mz.down(), mz.ancestor_count()), (0, 0, 0));
        assert!((mz.nominal_kinship() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn nominal_kinship_halves_with_each_degree() {
        for (cat, want) in [
            (Category::MZ, 0.5),
            (Category::FS, 0.25),
            (Category::PHS, 0.125),
            (Category::C1, 0.0625),
            (Category::C2, 0.015625),
        ] {
            assert!(
                (cat.nominal_kinship() - want).abs() < 1e-15,
                "{}",
                cat.code()
            );
        }
    }
}
