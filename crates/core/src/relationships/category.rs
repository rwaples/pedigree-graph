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
