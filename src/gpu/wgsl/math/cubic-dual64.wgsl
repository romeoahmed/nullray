/** Cubic roots and modulus differentiated before rounding to the evaluator precision. */
struct Dual64Elliptic {
  root: Dual64,
  scale: Dual64,
  parameter: Dual64,
  three_real: bool,
  valid: bool,
}

fn d64_elliptic(g2: Dual64, g3: Dual64) -> Dual64Elliptic {
  if (!d64_valid(g2) || !d64_valid(g3)) { return Dual64Elliptic(); }
  let primal = soft64_elliptic(g2.value,g3.value);
  if (!primal.valid) { return Dual64Elliptic(); }
  let zero = vec2u(0u);
  if (soft64_zero(g2.value) && soft64_zero(g3.value)) {
    if (!soft64_zero(g2.dx) || !soft64_zero(g2.dy) || !soft64_zero(g3.dx) || !soft64_zero(g3.dy)) { return Dual64Elliptic(); }
    return Dual64Elliptic(d64_constant(zero),d64_constant(zero),d64_constant(zero),false,true);
  }
  var isolated = primal.root;
  if (primal.three_real && (g3.value.y & 0x80000000u) == 0u) {
    isolated = soft64_add(primal.root,primal.scale);
  }
  // Implicit differentiation of 4 r³ - g2 r - g3 = 0 avoids acos at a close root pair.
  let slope = soft64_subtract(soft64_multiply(soft64_number(12.0),soft64_multiply(isolated,isolated)),g2.value);
  if (!soft64_valid(slope) || soft64_zero(slope)) { return Dual64Elliptic(); }
  let root = Dual64(isolated,
    soft64_divide(soft64_add(soft64_multiply(isolated,g2.dx),g3.dx),slope),
    soft64_divide(soft64_add(soft64_multiply(isolated,g2.dy),g3.dy),slope));
  let three = d64_constant(soft64_number(3.0));
  let square = d64_multiply(root,root);
  var lower = root;
  var scale: Dual64;
  var parameter: Dual64;
  if (primal.three_real) {
    let separation = d64_sqrt(d64_subtract(g2,d64_multiply(three,square)));
    if (!d64_valid(separation)) { return Dual64Elliptic(); }
    let left = d64_scale(d64_subtract(d64_negate(root),separation),-1);
    let right = d64_scale(d64_add(d64_negate(root),separation),-1);
    var middle = left;
    var upper = right;
    if (soft64_less(left.value,root.value)) { lower = left; middle = right; upper = root; }
    scale = d64_subtract(upper,lower);
    parameter = d64_divide(d64_subtract(middle,lower),scale);
  } else {
    scale = d64_sqrt(d64_subtract(d64_multiply(three,square),d64_scale(g2,-2)));
    parameter = d64_subtract(d64_constant(soft64_number(0.5)),d64_divide(d64_multiply(three,root),d64_scale(scale,2)));
  }
  return Dual64Elliptic(lower,scale,parameter,primal.three_real,
    d64_valid(lower) && d64_valid(scale) && d64_valid(parameter));
}
