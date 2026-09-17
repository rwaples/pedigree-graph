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

    /// Kinship degree: 0 for MZ, 1 for parent-offspring and full sibs, up to 5.
    pub fn degree(self) -> u8 {
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

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Whether the category has no roles, so a pair is stored `first < second`.
    ///
    /// The seven symmetric codes of the Python registry.  Every other category
    /// names a `first_role` and a `second_role`.
    pub fn symmetric(self) -> bool {
        matches!(
            self,
            Category::MZ
                | Category::FS
                | Category::MHS
                | Category::PHS
                | Category::C1
                | Category::H1C
                | Category::C2
        )
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
