//! # Crystal Lattice
//!
//! Crystallographic data structures for materials science.
//! Provides unit cells, Bravais lattice types, Miller indices,
//! crystal defects, and space group symmetry operations.

use std::f64::consts::{PI, FRAC_PI_2, FRAC_PI_3};

// ── unit_cell ───────────────────────────────────────────────────────────────

/// The fundamental repeating unit of a crystal lattice.
#[derive(Debug, Clone)]
pub struct UnitCell {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub alpha: f64, // radians
    pub beta: f64,
    pub gamma: f64,
}

impl UnitCell {
    pub fn cubic(a: f64) -> Self {
        Self { a, b: a, c: a, alpha: FRAC_PI_2, beta: FRAC_PI_2, gamma: FRAC_PI_2 }
    }

    pub fn tetragonal(a: f64, c: f64) -> Self {
        Self { a, b: a, c, alpha: FRAC_PI_2, beta: FRAC_PI_2, gamma: FRAC_PI_2 }
    }

    pub fn orthorhombic(a: f64, b: f64, c: f64) -> Self {
        Self { a, b, c, alpha: FRAC_PI_2, beta: FRAC_PI_2, gamma: FRAC_PI_2 }
    }

    pub fn hexagonal(a: f64, c: f64) -> Self {
        Self { a, b: a, c, alpha: FRAC_PI_2, beta: FRAC_PI_2, gamma: FRAC_PI_3 * 2.0 }
    }

    pub fn volume(&self) -> f64 {
        let ca = self.alpha.cos();
        let cb = self.beta.cos();
        let cg = self.gamma.cos();
        self.a * self.b * self.c * (1.0 - ca*ca - cb*cb - cg*cg + 2.0*ca*cb*cg).sqrt()
    }

    pub fn is_cubic(&self) -> bool {
        let eq = |a: f64, b: f64| (a - b).abs() < 1e-10;
        eq(self.a, self.b) && eq(self.b, self.c) &&
        eq(self.alpha, FRAC_PI_2) && eq(self.beta, FRAC_PI_2) && eq(self.gamma, FRAC_PI_2)
    }

    pub fn surface_area(&self) -> f64 {
        2.0 * (self.a * self.b + self.b * self.c + self.a * self.c)
    }

    pub fn fractional_to_cartesian(&self, fx: f64, fy: f64, fz: f64) -> (f64, f64, f64) {
        let x = self.a * fx;
        let y = self.b * fy * self.gamma.cos();
        let z = self.c * fz;
        (x, y, z)
    }

    pub fn lattice_parameters(&self) -> [f64; 6] {
        [self.a, self.b, self.c, self.alpha, self.beta, self.gamma]
    }
}

// ── bravais ─────────────────────────────────────────────────────────────────

/// The 14 Bravais lattice types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BravaisType {
    CubicP, CubicI, CubicF,
    TetragonalP, TetragonalI,
    OrthorhombicP, OrthorhombicC, OrthorhombicI, OrthorhombicF,
    HexagonalP,
    TrigonalR,
    MonoclinicP, MonoclinicC,
    TriclinicP,
}

impl BravaisType {
    pub fn all() -> Vec<BravaisType> {
        use BravaisType::*;
        vec![CubicP, CubicI, CubicF, TetragonalP, TetragonalI,
             OrthorhombicP, OrthorhombicC, OrthorhombicI, OrthorhombicF,
             HexagonalP, TrigonalR, MonoclinicP, MonoclinicC, TriclinicP]
    }

    pub fn crystal_system(&self) -> &'static str {
        match self {
            BravaisType::CubicP | BravaisType::CubicI | BravaisType::CubicF => "cubic",
            BravaisType::TetragonalP | BravaisType::TetragonalI => "tetragonal",
            BravaisType::OrthorhombicP | BravaisType::OrthorhombicC |
            BravaisType::OrthorhombicI | BravaisType::OrthorhombicF => "orthorhombic",
            BravaisType::HexagonalP => "hexagonal",
            BravaisType::TrigonalR => "trigonal",
            BravaisType::MonoclinicP | BravaisType::MonoclinicC => "monoclinic",
            BravaisType::TriclinicP => "triclinic",
        }
    }

    pub fn centering(&self) -> &'static str {
        match self {
            BravaisType::CubicP | BravaisType::TetragonalP | BravaisType::OrthorhombicP |
            BravaisType::HexagonalP | BravaisType::TrigonalR |
            BravaisType::MonoclinicP | BravaisType::TriclinicP => "P",
            BravaisType::CubicI | BravaisType::TetragonalI | BravaisType::OrthorhombicI => "I",
            BravaisType::CubicF | BravaisType::OrthorhombicF => "F",
            BravaisType::OrthorhombicC | BravaisType::MonoclinicC => "C",
        }
    }

    pub fn atoms_per_cell(&self) -> u32 {
        match self.centering() {
            "P" => 1,
            "I" => 2,
            "F" => 4,
            "C" => 2,
            _ => 1,
        }
    }

    pub fn multiplicity(&self) -> u32 {
        self.atoms_per_cell()
    }

    pub fn is_bravais_valid(&self) -> bool {
        Self::all().contains(self)
    }
}

// ── miller ──────────────────────────────────────────────────────────────────

/// Miller index (h, k, l) specifying a crystal plane.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Miller {
    pub h: i32,
    pub k: i32,
    pub l: i32,
}

impl Miller {
    pub fn new(h: i32, k: i32, l: i32) -> Self {
        Self { h, k, l }
    }

    pub fn zero() -> Self { Self::new(0, 0, 0) }

    pub fn is_valid(&self) -> bool {
        !(self.h == 0 && self.k == 0 && self.l == 0)
    }

    pub fn spacing(&self, cell: &UnitCell) -> f64 {
        if !self.is_valid() { return 0.0; }
        let h = self.h as f64;
        let k = self.k as f64;
        let l = self.l as f64;
        let inv_d2 = h*h / (cell.a*cell.a) + k*k / (cell.b*cell.b) + l*l / (cell.c*cell.c);
        if inv_d2 <= 0.0 { return f64::INFINITY; }
        1.0 / inv_d2.sqrt()
    }

    pub fn normal(&self) -> (f64, f64, f64) {
        let len = (self.h.pow(2) + self.k.pow(2) + self.l.pow(2)) as f64;
        if len == 0.0 { return (0.0, 0.0, 0.0); }
        let len = len.sqrt();
        (self.h as f64 / len, self.k as f64 / len, self.l as f64 / len)
    }

    pub fn angle_with(&self, other: &Miller) -> f64 {
        let dot = (self.h * other.h + self.k * other.k + self.l * other.l) as f64;
        let mag1 = (self.h.pow(2) + self.k.pow(2) + self.l.pow(2)) as f64;
        let mag2 = (other.h.pow(2) + other.k.pow(2) + other.l.pow(2)) as f64;
        if mag1 == 0.0 || mag2 == 0.0 { return 0.0; }
        let cos_a = dot / (mag1.sqrt() * mag2.sqrt());
        cos_a.clamp(-1.0, 1.0).acos()
    }

    pub fn is_parallel_to(&self, other: &Miller) -> bool {
        if !self.is_valid() || !other.is_valid() { return false; }
        self.h * other.k == self.k * other.h &&
        self.h * other.l == self.l * other.h
    }

    pub fn sum(&self) -> i32 {
        self.h + self.k + self.l
    }

    pub fn magnitude_squared(&self) -> i32 {
        self.h * self.h + self.k * self.k + self.l * self.l
    }

    pub fn zone_axis(&self, other: &Miller) -> Miller {
        Miller::new(
            self.k * other.l - self.l * other.k,
            self.l * other.h - self.h * other.l,
            self.h * other.k - self.k * other.h,
        )
    }
}

// ── defect ──────────────────────────────────────────────────────────────────

/// Crystal defects: point, line, and plane types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefectType {
    Vacancy,
    Interstitial,
    Substitutional,
    Frenkel,
    Schottky,
    EdgeDislocation,
    ScrewDislocation,
    MixedDislocation,
    GrainBoundary,
    StackingFault,
    TwinBoundary,
}

#[derive(Debug, Clone)]
pub struct Defect {
    pub defect_type: DefectType,
    pub position: (f64, f64, f64),
    pub energy: f64,
}

impl Defect {
    pub fn new(defect_type: DefectType, position: (f64, f64, f64), energy: f64) -> Self {
        Self { defect_type, position, energy }
    }

    pub fn is_point_defect(&self) -> bool {
        matches!(self.defect_type, DefectType::Vacancy | DefectType::Interstitial |
            DefectType::Substitutional | DefectType::Frenkel | DefectType::Schottky)
    }

    pub fn is_line_defect(&self) -> bool {
        matches!(self.defect_type, DefectType::EdgeDislocation |
            DefectType::ScrewDislocation | DefectType::MixedDislocation)
    }

    pub fn is_plane_defect(&self) -> bool {
        matches!(self.defect_type, DefectType::GrainBoundary |
            DefectType::StackingFault | DefectType::TwinBoundary)
    }

    pub fn category(&self) -> &'static str {
        if self.is_point_defect() { "point" }
        else if self.is_line_defect() { "line" }
        else { "plane" }
    }

    pub fn distance_to(&self, other: &Defect) -> f64 {
        let dx = self.position.0 - other.position.0;
        let dy = self.position.1 - other.position.1;
        let dz = self.position.2 - other.position.2;
        (dx*dx + dy*dy + dz*dz).sqrt()
    }

    pub fn with_energy(&self, energy: f64) -> Defect {
        Defect { energy, ..self.clone() }
    }

    pub fn at_origin(defect_type: DefectType, energy: f64) -> Self {
        Self::new(defect_type, (0.0, 0.0, 0.0), energy)
    }
}

// ── symmetry ────────────────────────────────────────────────────────────────

/// Space group symmetry operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryOp {
    Identity,
    Inversion,
    Rotation2,
    Rotation3,
    Rotation4,
    Rotation6,
    Mirror,
    Rotoinversion2,
    Rotoinversion3,
    Rotoinversion4,
    Rotoinversion6,
}

impl SymmetryOp {
    pub fn apply_to(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        match self {
            SymmetryOp::Identity => (x, y, z),
            SymmetryOp::Inversion => (-x, -y, -z),
            SymmetryOp::Rotation2 => (-x, -y, z),
            SymmetryOp::Rotation3 => {
                let c = (2.0 * PI / 3.0).cos();
                let s = (2.0 * PI / 3.0).sin();
                (x * c - y * s, x * s + y * c, z)
            }
            SymmetryOp::Rotation4 => (-y, x, z),
            SymmetryOp::Rotation6 => {
                let c = (PI / 3.0).cos();
                let s = (PI / 3.0).sin();
                (x * c - y * s, x * s + y * c, z)
            }
            SymmetryOp::Mirror => (x, y, -z),
            SymmetryOp::Rotoinversion2 => (x, y, -z),
            SymmetryOp::Rotoinversion3 => {
                let c = (2.0 * PI / 3.0).cos();
                let s = (2.0 * PI / 3.0).sin();
                (-(x * c - y * s), -(x * s + y * c), -z)
            }
            SymmetryOp::Rotoinversion4 => (y, -x, -z),
            SymmetryOp::Rotoinversion6 => {
                let c = (PI / 3.0).cos();
                let s = (PI / 3.0).sin();
                (-(x * c - y * s), -(x * s + y * c), -z)
            }
        }
    }

    pub fn order(&self) -> u32 {
        match self {
            SymmetryOp::Identity => 1,
            SymmetryOp::Inversion => 2,
            SymmetryOp::Rotation2 | SymmetryOp::Mirror => 2,
            SymmetryOp::Rotation3 => 3,
            SymmetryOp::Rotation4 => 4,
            SymmetryOp::Rotation6 => 6,
            SymmetryOp::Rotoinversion2 => 2,
            SymmetryOp::Rotoinversion3 => 6,
            SymmetryOp::Rotoinversion4 => 4,
            SymmetryOp::Rotoinversion6 => 6,
        }
    }

    pub fn is_rotation(&self) -> bool {
        matches!(self, SymmetryOp::Rotation2 | SymmetryOp::Rotation3 |
            SymmetryOp::Rotation4 | SymmetryOp::Rotation6)
    }

    pub fn is_improper(&self) -> bool {
        matches!(self, SymmetryOp::Inversion | SymmetryOp::Mirror |
            SymmetryOp::Rotoinversion2 | SymmetryOp::Rotoinversion3 |
            SymmetryOp::Rotoinversion4 | SymmetryOp::Rotoinversion6)
    }

    pub fn compose(&self, other: &SymmetryOp) -> impl Fn(f64, f64, f64) -> (f64, f64, f64) {
        let first = *self;
        let second = *other;
        move |x, y, z| {
            let (x1, y1, z1) = first.apply_to(x, y, z);
            second.apply_to(x1, y1, z1)
        }
    }

    pub fn determinant(&self) -> f64 {
        match self {
            SymmetryOp::Identity | SymmetryOp::Rotation2 | SymmetryOp::Rotation3 |
            SymmetryOp::Rotation4 | SymmetryOp::Rotation6 => 1.0,
            SymmetryOp::Inversion | SymmetryOp::Mirror |
            SymmetryOp::Rotoinversion2 | SymmetryOp::Rotoinversion3 |
            SymmetryOp::Rotoinversion4 | SymmetryOp::Rotoinversion6 => -1.0,
        }
    }
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod unit_cell_tests {
    use super::*;

    #[test]
    fn test_cubic() {
        let c = UnitCell::cubic(4.0);
        assert!((c.a - 4.0).abs() < 1e-10);
        assert!(c.is_cubic());
    }

    #[test]
    fn test_cubic_volume() {
        let c = UnitCell::cubic(2.0);
        assert!((c.volume() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_tetragonal() {
        let t = UnitCell::tetragonal(3.0, 5.0);
        assert!((t.a - 3.0).abs() < 1e-10);
        assert!((t.c - 5.0).abs() < 1e-10);
        assert!(!t.is_cubic());
    }

    #[test]
    fn test_orthorhombic() {
        let o = UnitCell::orthorhombic(2.0, 3.0, 4.0);
        assert!((o.volume() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_hexagonal() {
        let h = UnitCell::hexagonal(3.0, 5.0);
        assert!((h.gamma - (FRAC_PI_3 * 2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_surface_area() {
        let c = UnitCell::cubic(2.0);
        assert!((c.surface_area() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_fractional_to_cartesian() {
        let c = UnitCell::cubic(2.0);
        let (x, y, z) = c.fractional_to_cartesian(0.5, 0.5, 0.5);
        assert!((x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lattice_parameters() {
        let c = UnitCell::cubic(5.0);
        let p = c.lattice_parameters();
        assert_eq!(p.len(), 6);
        assert!((p[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_not_cubic() {
        let t = UnitCell::tetragonal(2.0, 3.0);
        assert!(!t.is_cubic());
    }
}

#[cfg(test)]
mod bravais_tests {
    use super::*;

    #[test]
    fn test_all_14() {
        assert_eq!(BravaisType::all().len(), 14);
    }

    #[test]
    fn test_cubic_systems() {
        assert_eq!(BravaisType::CubicP.crystal_system(), "cubic");
        assert_eq!(BravaisType::CubicI.crystal_system(), "cubic");
        assert_eq!(BravaisType::CubicF.crystal_system(), "cubic");
    }

    #[test]
    fn test_centering() {
        assert_eq!(BravaisType::CubicP.centering(), "P");
        assert_eq!(BravaisType::CubicI.centering(), "I");
        assert_eq!(BravaisType::CubicF.centering(), "F");
    }

    #[test]
    fn test_atoms_per_cell() {
        assert_eq!(BravaisType::CubicP.atoms_per_cell(), 1);
        assert_eq!(BravaisType::CubicI.atoms_per_cell(), 2);
        assert_eq!(BravaisType::CubicF.atoms_per_cell(), 4);
    }

    #[test]
    fn test_triclinic() {
        assert_eq!(BravaisType::TriclinicP.crystal_system(), "triclinic");
        assert_eq!(BravaisType::TriclinicP.centering(), "P");
    }

    #[test]
    fn test_monoclinic() {
        assert_eq!(BravaisType::MonoclinicP.crystal_system(), "monoclinic");
        assert_eq!(BravaisType::MonoclinicC.centering(), "C");
    }

    #[test]
    fn test_hexagonal() {
        assert_eq!(BravaisType::HexagonalP.crystal_system(), "hexagonal");
    }

    #[test]
    fn test_trigonal() {
        assert_eq!(BravaisType::TrigonalR.crystal_system(), "trigonal");
    }

    #[test]
    fn test_validity() {
        assert!(BravaisType::CubicP.is_bravais_valid());
    }

    #[test]
    fn test_orthorhombic_variants() {
        assert_eq!(BravaisType::OrthorhombicP.crystal_system(), "orthorhombic");
        assert_eq!(BravaisType::OrthorhombicI.crystal_system(), "orthorhombic");
        assert_eq!(BravaisType::OrthorhombicF.crystal_system(), "orthorhombic");
        assert_eq!(BravaisType::OrthorhombicC.crystal_system(), "orthorhombic");
    }
}

#[cfg(test)]
mod miller_tests {
    use super::*;

    #[test]
    fn test_new() {
        let m = Miller::new(1, 2, 3);
        assert_eq!(m.h, 1);
        assert_eq!(m.k, 2);
        assert_eq!(m.l, 3);
    }

    #[test]
    fn test_is_valid() {
        assert!(Miller::new(1, 0, 0).is_valid());
        assert!(!Miller::zero().is_valid());
    }

    #[test]
    fn test_spacing_cubic() {
        let cell = UnitCell::cubic(4.0);
        let m = Miller::new(1, 0, 0);
        assert!((m.spacing(&cell) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_spacing_110() {
        let cell = UnitCell::cubic(2.0);
        let m = Miller::new(1, 1, 0);
        let expected = 2.0 / 2.0_f64.sqrt();
        assert!((m.spacing(&cell) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_spacing_invalid() {
        let cell = UnitCell::cubic(2.0);
        assert_eq!(Miller::zero().spacing(&cell), 0.0);
    }

    #[test]
    fn test_normal() {
        let m = Miller::new(0, 0, 1);
        let (x, y, z) = m.normal();
        assert!((z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_angle_perpendicular() {
        let a = Miller::new(1, 0, 0);
        let b = Miller::new(0, 1, 0);
        assert!((a.angle_with(&b) - FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_angle_parallel() {
        let a = Miller::new(1, 0, 0);
        let b = Miller::new(2, 0, 0);
        assert!((a.angle_with(&b)).abs() < 1e-10);
    }

    #[test]
    fn test_is_parallel() {
        let a = Miller::new(1, 1, 0);
        let b = Miller::new(2, 2, 0);
        assert!(a.is_parallel_to(&b));
    }

    #[test]
    fn test_not_parallel() {
        let a = Miller::new(1, 0, 0);
        let b = Miller::new(0, 1, 0);
        assert!(!a.is_parallel_to(&b));
    }

    #[test]
    fn test_sum() {
        let m = Miller::new(1, 2, 3);
        assert_eq!(m.sum(), 6);
    }

    #[test]
    fn test_magnitude_squared() {
        let m = Miller::new(1, 1, 0);
        assert_eq!(m.magnitude_squared(), 2);
    }

    #[test]
    fn test_zone_axis() {
        let a = Miller::new(1, 0, 0);
        let b = Miller::new(0, 1, 0);
        let z = a.zone_axis(&b);
        assert_eq!(z.l, 1);
    }
}

#[cfg(test)]
mod defect_tests {
    use super::*;

    #[test]
    fn test_vacancy_is_point() {
        let d = Defect::at_origin(DefectType::Vacancy, 1.0);
        assert!(d.is_point_defect());
        assert!(!d.is_line_defect());
    }

    #[test]
    fn test_interstitial_is_point() {
        let d = Defect::at_origin(DefectType::Interstitial, 2.0);
        assert!(d.is_point_defect());
    }

    #[test]
    fn test_edge_dislocation_is_line() {
        let d = Defect::at_origin(DefectType::EdgeDislocation, 5.0);
        assert!(d.is_line_defect());
    }

    #[test]
    fn test_screw_dislocation_is_line() {
        let d = Defect::at_origin(DefectType::ScrewDislocation, 4.0);
        assert!(d.is_line_defect());
    }

    #[test]
    fn test_grain_boundary_is_plane() {
        let d = Defect::at_origin(DefectType::GrainBoundary, 10.0);
        assert!(d.is_plane_defect());
    }

    #[test]
    fn test_stacking_fault_is_plane() {
        let d = Defect::at_origin(DefectType::StackingFault, 3.0);
        assert!(d.is_plane_defect());
    }

    #[test]
    fn test_category() {
        assert_eq!(Defect::at_origin(DefectType::Vacancy, 1.0).category(), "point");
        assert_eq!(Defect::at_origin(DefectType::EdgeDislocation, 1.0).category(), "line");
        assert_eq!(Defect::at_origin(DefectType::TwinBoundary, 1.0).category(), "plane");
    }

    #[test]
    fn test_distance_to() {
        let a = Defect::new(DefectType::Vacancy, (0.0, 0.0, 0.0), 1.0);
        let b = Defect::new(DefectType::Vacancy, (3.0, 4.0, 0.0), 1.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_with_energy() {
        let d = Defect::at_origin(DefectType::Vacancy, 1.0);
        let d2 = d.with_energy(5.0);
        assert!((d2.energy - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_frenkel() {
        let d = Defect::at_origin(DefectType::Frenkel, 3.0);
        assert!(d.is_point_defect());
    }

    #[test]
    fn test_schottky() {
        let d = Defect::at_origin(DefectType::Schottky, 2.0);
        assert!(d.is_point_defect());
    }

    #[test]
    fn test_twin_boundary() {
        let d = Defect::at_origin(DefectType::TwinBoundary, 4.0);
        assert!(d.is_plane_defect());
    }
}

#[cfg(test)]
mod symmetry_tests {
    use super::*;

    #[test]
    fn test_identity() {
        let (x, y, z) = SymmetryOp::Identity.apply_to(1.0, 2.0, 3.0);
        assert!((x - 1.0).abs() < 1e-10);
        assert!((y - 2.0).abs() < 1e-10);
        assert!((z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_inversion() {
        let (x, y, z) = SymmetryOp::Inversion.apply_to(1.0, 2.0, 3.0);
        assert!((x + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation2() {
        let (x, y, z) = SymmetryOp::Rotation2.apply_to(1.0, 2.0, 3.0);
        assert!((x + 1.0).abs() < 1e-10);
        assert!((z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation4() {
        let (x, y, z) = SymmetryOp::Rotation4.apply_to(1.0, 0.0, 0.0);
        assert!((x).abs() < 1e-10);
        assert!((y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mirror() {
        let (x, y, z) = SymmetryOp::Mirror.apply_to(1.0, 2.0, 3.0);
        assert!((z + 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_identity_order() {
        assert_eq!(SymmetryOp::Identity.order(), 1);
    }

    #[test]
    fn test_rotation3_order() {
        assert_eq!(SymmetryOp::Rotation3.order(), 3);
    }

    #[test]
    fn test_rotation6_order() {
        assert_eq!(SymmetryOp::Rotation6.order(), 6);
    }

    #[test]
    fn test_is_rotation() {
        assert!(SymmetryOp::Rotation4.is_rotation());
        assert!(!SymmetryOp::Mirror.is_rotation());
    }

    #[test]
    fn test_is_improper() {
        assert!(SymmetryOp::Mirror.is_improper());
        assert!(!SymmetryOp::Rotation3.is_improper());
    }

    #[test]
    fn test_determinant_proper() {
        assert!((SymmetryOp::Rotation4.determinant() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_improper() {
        assert!((SymmetryOp::Mirror.determinant() + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compose() {
        let composed = SymmetryOp::Rotation2.compose(&SymmetryOp::Inversion);
        let (x, y, z) = composed(1.0, 0.0, 0.0);
        assert!((x - 1.0).abs() < 1e-10);
        assert!((y).abs() < 1e-10);
    }
}
