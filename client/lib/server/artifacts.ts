import { createHash } from 'node:crypto';

export function contentHash(data: Buffer) {
  return createHash('sha256').update(data).digest('hex');
}
export function pngData(encoded: string) {
  if (encoded.length > 4 * Math.ceil(8388608 / 3)) throw new Error('Oversized PNG');
  const data = Buffer.from(encoded, 'base64');
  if (data.length < 33 || data.length > 8388608 || !data.subarray(0, 8).equals(Buffer.from('89504e470d0a1a0a', 'hex')) || data.toString('ascii', 12, 16) !== 'IHDR') throw new Error('Invalid PNG');
  const width = data.readUInt32BE(16), height = data.readUInt32BE(20);
  if (!width || !height || width > 4096 || height > 4096) throw new Error('Invalid PNG dimensions');
  return data;
}
