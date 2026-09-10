/** High-precision polynomial values, invariants and derivatives at the observer. */
struct Dual64Quartic {
  values: array<Dual64,9>,
  elliptic: Dual64Elliptic,
  valid: bool,
}

fn d64_prepare_quartic(c: array<Dual64,5>, x: Dual64, velocity: Dual64) -> Dual64Quartic {
  let two = d64_constant(soft64_number(2.0));
  let three = d64_constant(soft64_number(3.0));
  let four = d64_constant(soft64_number(4.0));
  let six = d64_constant(soft64_number(6.0));
  let twelve = d64_constant(soft64_number(12.0));
  let sixteen = d64_constant(soft64_number(16.0));
  let twenty_four = d64_constant(soft64_number(24.0));
  let forty_eight = d64_constant(soft64_number(48.0));
  let c22 = d64_multiply(c[2],c[2]);
  let g2 = d64_add(d64_subtract(d64_multiply(c[4],c[0]),d64_scale(d64_multiply(c[3],c[1]),-2)),d64_divide(c22,twelve));
  let g3 = d64_subtract(d64_subtract(d64_subtract(d64_add(
    d64_divide(d64_multiply(d64_multiply(c[4],c[2]),c[0]),six),
    d64_divide(d64_multiply(d64_multiply(c[3],c[2]),c[1]),forty_eight)),
    d64_divide(d64_multiply(c22,c[2]),d64_constant(soft64_number(216.0)))),
    d64_divide(d64_multiply(d64_multiply(c[4],c[1]),c[1]),sixteen)),
    d64_divide(d64_multiply(d64_multiply(c[3],c[3]),c[0]),sixteen));
  var elliptic: Dual64Elliptic;
  let quadratic_family = soft64_zero(c[3].value) && soft64_zero(c[3].dx) && soft64_zero(c[3].dy) &&
    soft64_zero(c[4].value) && soft64_zero(c[4].dx) && soft64_zero(c[4].dy);
  if (quadratic_family && d64_valid(c[2]) && !soft64_zero(c[2].value)) {
    // For f(x)=c0+c1*x+c2*x² the elliptic cubic has an exact repeated pair.
    // Differentiate its constrained quadratic family directly; differentiating
    // the square root of the vanishing generic discriminant is undefined.
    if (soft64_less(c[2].value,vec2u(0u))) {
      elliptic = Dual64Elliptic(d64_divide(c[2],twelve),d64_scale(d64_negate(c[2]),-2),
        d64_constant(vec2u(0u)),true,true);
    } else {
      elliptic = Dual64Elliptic(d64_negate(d64_divide(c[2],six)),d64_scale(c[2],-2),
        d64_constant(soft64_number(1.0)),true,true);
    }
  } else {
    elliptic = d64_elliptic(g2,g3);
  }
  if (!elliptic.valid) { return Dual64Quartic(); }
  var f = c[4];
  for (var index = 4u; index > 0u; index--) { f = d64_add(d64_multiply(f,x),c[index-1u]); }
  let first = d64_add(d64_multiply(d64_add(d64_multiply(d64_add(d64_multiply(d64_multiply(four,c[4]),x),d64_multiply(three,c[3])),x),d64_multiply(two,c[2])),x),c[1]);
  let second = d64_add(d64_multiply(d64_add(d64_multiply(d64_multiply(twelve,c[4]),x),d64_multiply(six,c[3])),x),d64_multiply(two,c[2]));
  let third = d64_add(d64_multiply(d64_multiply(twenty_four,c[4]),x),d64_multiply(six,c[3]));
  let fourth = d64_multiply(twenty_four,c[4]);
  return Dual64Quartic(array<Dual64,9>(x,velocity,f,first,second,third,fourth,g2,g3),elliptic,true);
}

fn d64_round(status: ptr<function,bool>, basis: array<Soft64,2>, value: Dual64) -> vec3f {
  let along = soft64_add(soft64_multiply(basis[0],value.dx),soft64_multiply(basis[1],value.dy));
  let across = soft64_subtract(soft64_multiply(basis[0],value.dy),soft64_multiply(basis[1],value.dx));
  let bits = vec3u(soft64_to_f32_bits(value.value),soft64_to_f32_bits(along),soft64_to_f32_bits(across));
  if (any((bits & vec3u(0x7f800000u)) == vec3u(0x7f800000u))) { return d_fail(status); }
  return bitcast<vec3f>(bits);
}

fn d64_round_quartic(status: ptr<function,bool>, basis: array<Soft64,2>, prepared: Dual64Quartic) -> DifferentialQuartic {
  if (!prepared.valid) { *status = false; return DifferentialQuartic(); }
  var values: array<vec3f,9>;
  for (var i=0u;i<9u;i++) { values[i] = d64_round(status,basis,prepared.values[i]); }
  let e = prepared.elliptic;
  let root = d64_round(status,basis,e.root);
  let scale = d64_round(status,basis,e.scale);
  let parameter = d64_round(status,basis,e.parameter);
  let complement = d64_round(status,basis,d64_subtract(d64_constant(soft64_number(1.0)),e.parameter));
  return DifferentialQuartic(values[0],values[1],values[2],values[3],values[4],values[5],values[6],
    DifferentialElliptic(values[7],values[8],root,scale,parameter,complement,e.three_real,d_prepare_jacobi_pair(status,parameter,complement)));
}
