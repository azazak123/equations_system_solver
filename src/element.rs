use std::{
    iter::Sum,
    ops::{Add, Mul, Sub},
};

use num::{traits::Pow, One, Signed, Zero};

pub trait Element:
    Mul<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Sum
    + Clone
    + Zero
    + Copy
    + One
    + Signed
    + Pow<u8, Output = Self>
    + PartialOrd
{
}

macro_rules! el_impl {
    ($t:ty) => {
        impl Element for $t {}
    };
}

el_impl!(isize);
el_impl!(i8);
el_impl!(i16);
el_impl!(i32);
el_impl!(i64);
el_impl!(i128);

el_impl!(f32);
el_impl!(f64);
