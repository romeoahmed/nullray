import fieldSource from "../wgsl/sources/field.wgsl?raw";
import geometry from "../wgsl/geodesics/geometry.wgsl?raw";
import geodesics from "../wgsl/geodesics/orbit.wgsl?raw";
import medium from "../wgsl/geodesics/medium.wgsl?raw";
import barriers from "../wgsl/geodesics/barriers.wgsl?raw";
import radiation from "../wgsl/imaging/radiation.wgsl?raw";
import structure from "../wgsl/imaging/structure.wgsl?raw";
import frameSource from "../wgsl/imaging/frame.wgsl?raw";
import transfer from "../wgsl/imaging/transfer.wgsl?raw";
import matter from "../wgsl/imaging/matter.wgsl?raw";
import trace from "../wgsl/geodesics/trace.wgsl?raw";
import stellarSource from "../wgsl/sources/stars.wgsl?raw";
import starsPass from "../wgsl/passes/stars.wgsl?raw";
import imaging from "../wgsl/passes/image.wgsl?raw";

export const opticalSource = `${geometry}\n${geodesics}\n${medium}\n${barriers}\n${radiation}\n${fieldSource}\n${structure}\n${frameSource}\n${transfer}\n${matter}\n${trace}\n${imaging}`;
export const starsSource = `${frameSource}\n${radiation}\n${stellarSource}\n${starsPass}`;
