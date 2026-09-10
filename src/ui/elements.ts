/** Check static markup once, so event handlers operate on concrete element types. */
export function element<T extends Element>(
  root: ParentNode,
  selector: string,
  type: { new (): T },
): T {
  const value = root.querySelector(selector);
  if (!(value instanceof type)) {
    throw new Error(`Missing ${type.name}: ${selector}`);
  }
  return value;
}
