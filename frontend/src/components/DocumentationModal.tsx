import { useEffect, useRef } from 'react';

interface DocumentationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const methods = [
  ['canvas.set_color(r, g, b, a=1.0)', 'Set the drawing color using values from 0 to 1.'],
  ['canvas.move_brush_to(x, y)', 'Move to the starting point without drawing.'],
  ['canvas.draw_line_to(x, y)', 'Draw from the current brush position to a new point.'],
  ['canvas.draw_circle(x, y, radius, fill=False)', 'Draw an outlined or filled circle.'],
  ['canvas.draw_rectangle(x, y, width, height, fill=False)', 'Draw an outlined or filled rectangle.'],
  ['canvas.set_line_width(width)', 'Set the stroke width.'],
  ['canvas.fill_background(r, g, b)', 'Fill the canvas background.'],
  ['canvas.draw_polygon(points, fill=False)', 'Draw a polygon from at least three (x, y) points.'],
  ['canvas.draw_text(x, y, text, font_size=16)', 'Draw text at the given baseline position.'],
] as const;

export const DocumentationModal = ({ isOpen, onClose }: DocumentationModalProps) => {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (isOpen && dialog && !dialog.open) dialog.showModal();
    return () => {
      if (dialog?.open) dialog.close();
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const close = () => {
    dialogRef.current?.close();
    onClose();
  };

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="artcanvas-docs-title"
      className="w-full max-w-3xl max-h-[80vh] overflow-y-auto rounded-lg bg-base-100 p-8 backdrop:bg-black/50"
      onCancel={(event) => {
        event.preventDefault();
        close();
      }}
    >
      <button
        type="button"
        onClick={close}
        className="btn btn-sm btn-circle absolute right-4 top-4"
        aria-label="Close documentation"
      >
        x
      </button>
      <h4 id="artcanvas-docs-title" className="font-bold mb-6 text-2xl">ArtCanvas Documentation</h4>
      <p className="text-gray-600 mb-4">Use these commands to create a drawing on the 1920 x 1200 canvas.</p>

      <ul className="space-y-4">
        {methods.map(([signature, description]) => (
          <li key={signature} className="flex flex-col gap-1">
            <code className="bg-base-200 px-3 py-2 rounded-lg font-mono">{signature}</code>
            <span className="text-sm text-gray-600 ml-4">{description}</span>
          </li>
        ))}
      </ul>

      <div className="mt-8">
        <h5 className="font-bold mb-4">Example Code</h5>
        <pre className="bg-base-200 p-4 rounded-lg overflow-x-auto">
          <code className="text-sm">{`from artcanvas import ArtCanvas

with ArtCanvas() as canvas:
    canvas.fill_background(1, 1, 1)
    canvas.set_color(1, 0, 0, 0.8)
    canvas.set_line_width(3)

    # A line needs an explicit starting point.
    canvas.move_brush_to(100, 100)
    canvas.draw_line_to(500, 300)

    canvas.draw_circle(400, 200, 50, fill=True)
    points = [(800, 150), (900, 150), (850, 250)]
    canvas.draw_polygon(points, fill=True)

    canvas.set_color(0, 0, 0)
    canvas.draw_text(100, 600, "Hello World!", font_size=24)`}</code>
        </pre>
      </div>
    </dialog>
  );
};
