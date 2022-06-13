//! A collection of primitive SDFs that may the modified using functions in `mods`
//! or combined using functions in `ops`. Note that these primitives are always
//! centered around the origin and that you must transform the point you are sampling
//! into 'primitive-local' space. Functions are provided in `mods` to do this easier.
//!
//! Also note that while all translation and rotation transformations of the input point
//! will work properly, scaling modifies the Euclidian space and therefore does not work
//! normally. Utility function shave been provided to `Translate`, `Rotate`, and `Scale`
//! in the `SDF` trait and the `mods` module.
use palette::LinSrgba;

use crate::mathtypes::*;
use crate::SDF;
use std::marker::PhantomData;
use std::ops::*;

/// A sphere centered at origin with a radius.
#[derive(Clone, Copy, Debug)]
pub struct Sphere<T> {
    pub radius: T,
    pub colour: LinSrgba,
}

impl<T> Sphere<T> {
    pub fn new(radius: T, colour: LinSrgba) -> Self {
        Sphere { radius, colour }
    }
}

impl<T, V> SDF<T, V> for Sphere<T>
where
    T: Sub<T, Output = T> + Copy,
    V: Vec3<T>,
{
    fn dist(&self, p: V) -> T {
        p.magnitude() - self.radius
    }

    fn colour(&self, _p: V) -> LinSrgba {
        self.colour
    }
}

/// A box centered at origin with axis-aligned dimensions.
#[derive(Clone, Copy, Debug)]
pub struct Box<V, D> {
    pub dims: V,
    pub colour: LinSrgba,
    _pd: PhantomData<D>,
}

impl<V, D> Box<V, D> {
    pub fn new(dims: V, colour: LinSrgba) -> Self {
        Box {
            dims,
            colour,
            _pd: PhantomData,
        }
    }
}

impl<T, V> SDF<T, V> for Box<V, Dim3D>
where
    T: Add<T, Output = T> + MaxMin + Zero + Copy,
    V: Vec3<T> + Copy,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let d = p.abs() - self.dims;
        d.max(V::zero()).magnitude() + d.y().max(d.z()).max(d.x()).min(T::zero())
    }

    fn colour(&self, _p: V) -> LinSrgba {
        self.colour
    }
}

impl<T, V> SDF<T, V> for Box<V, Dim2D>
where
    T: Add<T, Output = T> + MaxMin + Zero + Copy,
    V: Vec2<T> + Copy,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let d = p.abs() - self.dims;
        d.max(V::zero()).magnitude() + d.y().max(d.x()).min(T::zero())
    }

    fn colour(&self, _p: V) -> LinSrgba {
        self.colour
    }
}

/// A circle centered at origin with a radius.
#[derive(Clone, Copy, Debug)]
pub struct Circle<T> {
    pub radius: T,
    pub colour: LinSrgba,
}

impl<T> Circle<T> {
    pub fn new(radius: T, colour: LinSrgba) -> Self {
        Circle { radius, colour }
    }
}

impl<T, V> SDF<T, V> for Circle<T>
where
    T: Sub<T, Output = T> + Copy,
    V: Vec2<T>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        p.magnitude() - self.radius
    }

    fn colour(&self, _p: V) -> LinSrgba {
        self.colour
    }
}

/// A torus that sits on the XZ plane. Thickness is the radius of
/// the wrapped cylinder while radius is the radius of the donut
/// shape.
#[derive(Clone, Copy, Debug)]
pub struct Torus<T> {
    pub radius: T,
    pub thickness: T,
    pub colour: LinSrgba,
}

impl<T> Torus<T> {
    pub fn new(radius: T, thickness: T, colour: LinSrgba) -> Self {
        Torus { radius, thickness, colour }
    }
}

impl<T, V> SDF<T, V> for Torus<T>
where
    T: Sub<T, Output = T> + Copy,
    V: Vec3<T>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let q = V::Vec2::new(
            V::Vec2::new(p.x(), p.z()).magnitude() - self.thickness,
            p.y(),
        );
        q.magnitude() - self.radius
    }

    fn colour(&self, _p: V) -> LinSrgba {
        self.colour
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// An infinite cylinder extending along an axis.
#[derive(Clone, Copy, Debug)]
pub struct Cylinder<T> {
    pub radius: T,
    pub axis: Axis,
    pub colour: LinSrgba,
}

impl<T> Cylinder<T> {
    pub fn new(radius: T, axis: Axis, colour: LinSrgba) -> Self {
        Cylinder { radius, axis, colour }
    }
}

impl<T, V> SDF<T, V> for Cylinder<T>
where
    T: Sub<T, Output = T> + Copy,
    V: Vec3<T>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let (a, b) = match self.axis {
            Axis::X => (p.y(), p.z()),
            Axis::Y => (p.x(), p.z()),
            Axis::Z => (p.x(), p.y()),
        };
        V::Vec2::new(a, b).magnitude() - self.radius
    }

    fn colour(&self, _p: V) -> LinSrgba {
        self.colour
    }
}

/// A capped cylinder extending along an axis.
#[derive(Clone, Copy, Debug)]
pub struct CappedCylinder<T> {
    pub radius: T,
    pub height: T,
    pub axis: Axis,
    pub colour: LinSrgba,
}

impl<T> CappedCylinder<T> {
    pub fn new(radius: T, height: T, axis: Axis, colour: LinSrgba) -> Self {
        CappedCylinder {
            radius,
            height,
            axis,
            colour,
        }
    }
}

impl<T, V> SDF<T, V> for CappedCylinder<T>
where
    T: Sub<T, Output = T> + Add<T, Output = T> + Zero + MaxMin + Copy,
    V: Vec3<T>,
{
    #[inline]
    #[allow(clippy::many_single_char_names)]
    fn dist(&self, p: V) -> T {
        let (a, b, c) = match self.axis {
            Axis::X => (p.y(), p.z(), p.x()),
            Axis::Y => (p.x(), p.z(), p.y()),
            Axis::Z => (p.x(), p.y(), p.z()),
        };
        let d = V::Vec2::new(V::Vec2::new(a, b).magnitude(), c).abs()
            - V::Vec2::new(self.radius, self.height);
        d.x().max(d.y()).min(T::zero()) + d.max(V::Vec2::zero()).magnitude()
    }

    fn colour(&self, _p: V) -> LinSrgba {
        self.colour
    }
}

/// A capsule extending from `a` to `b` with thickness `thickness`.
#[derive(Clone, Copy, Debug)]
pub struct Line<T, V> {
    pub a: V,
    pub b: V,
    pub thickness: T,
    pub colour: LinSrgba,
}

impl<T, V> Line<T, V> {
    pub fn new(a: V, b: V, thickness: T, colour: LinSrgba) -> Self {
        Line { a, b, thickness, colour }
    }
}

impl<T, V> SDF<T, V> for Line<T, V>
where
    T: Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T> + Zero + One + Clamp + Copy,
    V: Vec<T> + Copy,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let pa = p - self.a;
        let ba = self.b - self.a;
        let t = pa.dot(ba) / ba.dot(ba);
        let h = t.clamp(T::zero(), T::one());
        (pa - (ba * h)).magnitude() - self.thickness
    }

    fn colour(&self, _p: V) -> LinSrgba {
        self.colour
    }
}
