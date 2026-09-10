/** Integer-precision path preparation, rounded only at the f32 evaluator boundary. */
struct Prepared64 {
  path: QuarticPath,
  valid: bool,
}

struct Elliptic64 {
  root: Soft64,
  scale: Soft64,
  parameter: Soft64,
  three_real: bool,
  valid: bool,
}

fn soft64_ceil_divide(value: i32, divisor: i32) -> i32 {
  if (value >= 0) { return (value + divisor - 1) / divisor; }
  return value / divisor;
}

/**
 * Refine the isolated signed root of 4 x³ - g2 x - g3. Scaling keeps the root
 * solve dimensionless. The remaining quadratic gives the close root pair without
 * applying native f32 acos to a nearly repeated-root discriminant.
 */
fn soft64_elliptic(g2: Soft64, g3: Soft64) -> Elliptic64 {
  let zero = vec2u(0u);
  let one = soft64_number(1.0);
  let three = soft64_number(3.0);
  let four = soft64_number(4.0);
  let twelve = soft64_number(12.0);
  if (!soft64_valid(g2) || !soft64_valid(g3)) { return Elliptic64(); }
  if (soft64_zero(g2) && soft64_zero(g3)) { return Elliptic64(zero,zero,zero,false,true); }
  var power = -1024;
  if (!soft64_zero(g2)) { power = soft64_ceil_divide(i32((g2.y >> 20u) & 2047u) - 1022, 2); }
  if (!soft64_zero(g3)) { power = max(power, soft64_ceil_divide(i32((g3.y >> 20u) & 2047u) - 1022, 3)); }
  let scaled_g2 = soft64_scale(g2, -2 * power);
  let scaled_g3 = soft64_scale(soft64_abs(g3), -3 * power);
  if (!soft64_valid(scaled_g2) || !soft64_valid(scaled_g3)) { return Elliptic64(); }
  var root = one;
  var converged = false;
  for (var iteration = 0u; iteration < 32u; iteration++) {
    let square = soft64_multiply(root, root);
    let slope = soft64_subtract(soft64_multiply(twelve,square), scaled_g2);
    let residual = soft64_subtract(soft64_multiply(root,soft64_subtract(soft64_multiply(four,square),scaled_g2)),scaled_g3);
    if (!soft64_valid(slope) || !soft64_valid(residual) || !soft64_less(zero,slope)) { return Elliptic64(); }
    let correction = soft64_divide(residual,slope);
    let next = soft64_subtract(root,correction);
    if (!soft64_valid(next) || soft64_less(next,zero)) { return Elliptic64(); }
    root = next;
    // Relative last-step criterion; a small correction alone is not a root certificate.
    if (soft64_zero(correction) || soft64_less(soft64_abs(correction),soft64_scale(root,-50))) {
      converged = true;
      break;
    }
  }
  if (!converged) { return Elliptic64(); }
  if ((g3.y & 0x80000000u) != 0u) { root = soft64_negate(root); }
  let square = soft64_multiply(root,root);
  let difference = soft64_subtract(scaled_g2,soft64_multiply(three,square));
  if (!soft64_valid(difference)) { return Elliptic64(); }
  if (!soft64_less(difference,zero)) {
    let separation = soft64_sqrt(difference);
    let left = soft64_scale(soft64_subtract(soft64_negate(root),separation),-1);
    let right = soft64_scale(soft64_add(soft64_negate(root),separation),-1);
    var lower = root;
    var middle = left;
    var upper = right;
    if (soft64_less(left,root)) { lower = left; middle = right; upper = root; }
    let scale = soft64_subtract(upper,lower);
    let parameter = soft64_divide(soft64_subtract(middle,lower),scale);
    return Elliptic64(soft64_scale(lower,power),soft64_scale(scale,power),parameter,true,
      soft64_valid(parameter) && !soft64_less(parameter,zero) && !soft64_less(one,parameter));
  }
  let scale = soft64_sqrt(soft64_subtract(soft64_multiply(three,square),soft64_scale(scaled_g2,-2)));
  let parameter = soft64_subtract(soft64_scale(one,-1),soft64_divide(soft64_multiply(three,root),soft64_multiply(four,scale)));
  return Elliptic64(soft64_scale(root,power),soft64_scale(scale,power),parameter,false,
    soft64_valid(parameter) && !soft64_less(parameter,zero) && !soft64_less(one,parameter));
}

fn soft64_prepare_quartic(c: array<Soft64,5>, x: Soft64, velocity: Soft64) -> Prepared64 {
  let two = soft64_number(2.0);
  let three = soft64_number(3.0);
  let four = soft64_number(4.0);
  let six = soft64_number(6.0);
  let twelve = soft64_number(12.0);
  let sixteen = soft64_number(16.0);
  let twenty_four = soft64_number(24.0);
  let forty_eight = soft64_number(48.0);
  let c22 = soft64_multiply(c[2],c[2]);
  let g2 = soft64_add(soft64_subtract(soft64_multiply(c[4],c[0]),soft64_scale(soft64_multiply(c[3],c[1]),-2)),soft64_divide(c22,twelve));
  let g3 = soft64_subtract(soft64_subtract(soft64_subtract(soft64_add(
    soft64_divide(soft64_multiply(soft64_multiply(c[4],c[2]),c[0]),six),
    soft64_divide(soft64_multiply(soft64_multiply(c[3],c[2]),c[1]),forty_eight)),
    soft64_divide(soft64_multiply(c22,c[2]),soft64_number(216.0))),
    soft64_divide(soft64_multiply(soft64_multiply(c[4],c[1]),c[1]),sixteen)),
    soft64_divide(soft64_multiply(soft64_multiply(c[3],c[3]),c[0]),sixteen));
  var elliptic: Elliptic64;
  if (soft64_zero(c[3]) && soft64_zero(c[4]) && soft64_valid(c[2]) && !soft64_zero(c[2])) {
    // Exact quadratic potentials have a repeated elliptic root for every c0/c1.
    // Preserve their trigonometric or hyperbolic limit before invariant rounding.
    if (soft64_less(c[2],vec2u(0u))) {
      elliptic = Elliptic64(soft64_divide(c[2],twelve),soft64_scale(soft64_negate(c[2]),-2),vec2u(0u),true,true);
    } else {
      elliptic = Elliptic64(soft64_negate(soft64_divide(c[2],six)),soft64_scale(c[2],-2),soft64_number(1.0),true,true);
    }
  } else {
    elliptic = soft64_elliptic(g2,g3);
  }
  if (!elliptic.valid) { return Prepared64(); }
  var f = c[4];
  for (var index = 4u; index > 0u; index--) { f = soft64_add(soft64_multiply(f,x),c[index-1u]); }
  let first = soft64_add(soft64_multiply(soft64_add(soft64_multiply(soft64_add(soft64_multiply(soft64_multiply(four,c[4]),x),soft64_multiply(three,c[3])),x),soft64_multiply(two,c[2])),x),c[1]);
  let second = soft64_add(soft64_multiply(soft64_add(soft64_multiply(soft64_multiply(twelve,c[4]),x),soft64_multiply(six,c[3])),x),soft64_multiply(two,c[2]));
  let third = soft64_add(soft64_multiply(soft64_multiply(twenty_four,c[4]),x),soft64_multiply(six,c[3]));
  let fourth = soft64_multiply(twenty_four,c[4]);
  let values = array<Soft64,13>(x,velocity,f,first,second,third,fourth,g2,g3,elliptic.root,elliptic.scale,elliptic.parameter,soft64_subtract(soft64_number(1.0),elliptic.parameter));
  var converted: array<f32,13>;
  for (var index = 0u; index < 13u; index++) {
    let bits = soft64_to_f32_bits(values[index]);
    if ((bits & 0x7f800000u) == 0x7f800000u) { return Prepared64(); }
    converted[index] = bitcast<f32>(bits);
  }
  return Prepared64(QuarticPath(converted[0],converted[1],converted[2],converted[3],converted[4],converted[5],converted[6],
    Elliptic(converted[7],converted[8],converted[9],converted[10],converted[11],converted[12],elliptic.three_real)),true);
}

/** Radial forbidden intervals classified before coefficient rounding near a critical root. */
fn soft64_radial_interval(c: array<Soft64,5>, lower: Soft64, upper: Soft64) -> u32 {
  let zero = vec2u(0u);
  let a = soft64_scale(c[4],2);
  let b = soft64_multiply(soft64_number(3.0),c[3]);
  let constant = soft64_scale(c[2],1);
  var roots: array<Soft64,2>;
  var count = 0u;
  if (soft64_zero(a)) {
    if (!soft64_zero(b)) { roots[0] = soft64_divide(soft64_negate(constant),b); count = 1u; }
  } else {
    let discriminant = soft64_subtract(soft64_multiply(b,b),soft64_scale(soft64_multiply(a,constant),2));
    if (!soft64_valid(discriminant)) { return 2u; }
    if (!soft64_less(discriminant,zero)) {
      var root = soft64_sqrt(discriminant);
      if (soft64_less(b,zero)) { root = soft64_negate(root); }
      let term = soft64_scale(soft64_negate(soft64_add(b,root)),-1);
      if (!soft64_zero(term)) {
        roots[0] = soft64_divide(term,a);
        roots[1] = soft64_divide(constant,term);
        count = 2u;
      }
    }
  }
  var blocked = 0u;
  for (var index=0u;index<count;index++) {
    let u = roots[index];
    if (!soft64_valid(u)) { return 2u; }
    if (soft64_less(lower,u) && soft64_less(u,upper)) {
      let u2 = soft64_multiply(u,u);
      let value = soft64_add(c[0],soft64_multiply(u2,soft64_add(c[2],soft64_multiply(u,soft64_add(c[3],soft64_multiply(c[4],u))))));
      let magnitude = soft64_add(soft64_abs(c[0]),soft64_multiply(u2,soft64_add(soft64_abs(c[2]),soft64_multiply(u,soft64_add(soft64_abs(c[3]),soft64_multiply(soft64_abs(c[4]),u))))));
      if (!soft64_valid(value) || !soft64_valid(magnitude)) { return 2u; }
      // Sixteen binary64 epsilons times the absolute polynomial sum retain an explicit ambiguous band.
      let uncertainty = soft64_scale(magnitude,-48);
      if (!soft64_less(uncertainty,soft64_abs(value))) { return 2u; }
      if (soft64_less(value,zero)) { blocked = 1u; }
    }
  }
  return blocked;
}

/** Same exterior interval convention as radial_destination, evaluated with the prepared coefficients. */
fn soft64_radial_destination(c: array<Soft64,5>, x: Soft64, velocity: Soft64, horizon: Soft64) -> u32 {
  let zero = vec2u(0u);
  for (var index=0u;index<5u;index++) { if (!soft64_valid(c[index])) { return 0u; } }
  if (!soft64_zero(c[1]) || !soft64_valid(x) || soft64_less(x,zero) || !soft64_less(x,horizon)) { return 0u; }
  let first = soft64_multiply(x,soft64_add(soft64_scale(c[2],1),soft64_multiply(x,soft64_add(soft64_multiply(soft64_number(3.0),c[3]),soft64_scale(soft64_multiply(c[4],x),2)))));
  if (!soft64_valid(first) || !soft64_valid(velocity) || !soft64_valid(horizon)) { return 0u; }
  if (soft64_zero(velocity)) {
    let magnitude = soft64_multiply(x,soft64_add(soft64_scale(soft64_abs(c[2]),1),soft64_multiply(x,soft64_add(soft64_multiply(soft64_number(3.0),soft64_abs(c[3])),soft64_scale(soft64_multiply(soft64_abs(c[4]),x),2)))));
    if (!soft64_less(soft64_scale(magnitude,-48),soft64_abs(first))) { return 0u; }
  }
  let inward = soft64_less(zero,velocity) || (soft64_zero(velocity) && soft64_less(zero,first));
  let forward = soft64_radial_interval(c,select(zero,x,inward),select(x,horizon,inward));
  if (forward == 0u) { return select(1u,2u,inward); }
  if (forward == 2u) { return 0u; }
  let backward = soft64_radial_interval(c,select(x,zero,inward),select(horizon,x,inward));
  if (backward != 0u) { return 0u; }
  return select(2u,1u,inward);
}
