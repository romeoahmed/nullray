/**
 * Resolve required static markup and narrow it with a runtime constructor check.
 *
 * @returns The first matching element, borrowed from the caller's DOM tree.
 * @throws Error - If it is absent or has the wrong element type.
 */
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
