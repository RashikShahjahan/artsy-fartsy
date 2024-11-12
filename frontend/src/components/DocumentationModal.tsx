interface DocumentationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DocumentationModal = ({ isOpen, onClose }: DocumentationModalProps) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-base-100 p-8 rounded-lg max-w-3xl w-full max-h-[80vh] overflow-y-auto relative">
        <button 
          onClick={onClose}
          className="btn btn-sm btn-circle absolute right-4 top-4"
        >
          ✕
        </button>
        <h4 className="font-bold mb-6 text-2xl">ArtCanvas Documentation</h4>
        <div className="space-y-4">
                        <p className="text-gray-600 mb-4">Use these commands to create your artwork:</p>
                        <ul className="space-y-4">
                          <li className="flex flex-col gap-1">
                            <code className="bg-base-200 px-3 py-2 rounded-lg font-mono">canvas.set_color(r, g, b, a=1.0)</code>
                            <span className="text-sm text-gray-600 ml-4">Set drawing color (values 0-1)</span>
                          </li>
                          <li className="flex flex-col gap-1">
                            <code className="bg-base-200 px-3 py-2 rounded-lg font-mono">canvas.draw_line_to(x, y)</code>
                            <span className="text-sm text-gray-600 ml-4">Draw line to specified point</span>
                          </li>
                          <li className="flex flex-col gap-1">
                            <code className="bg-base-200 px-3 py-2 rounded-lg font-mono">canvas.draw_circle(x, y, radius, fill=False)</code>
                            <span className="text-sm text-gray-600 ml-4">Draw circle</span>
                          </li>
                          <li className="flex flex-col gap-1">
                            <code className="bg-base-200 px-3 py-2 rounded-lg font-mono">canvas.draw_rectangle(x, y, width, height, fill=False)</code>
                            <span className="text-sm text-gray-600 ml-4">Draw rectangle</span>
                          </li>
                          <li className="flex flex-col gap-1">
                            <code className="bg-base-200 px-3 py-2 rounded-lg font-mono">canvas.set_line_width(width)</code>
                            <span className="text-sm text-gray-600 ml-4">Set stroke width</span>
                          </li>
                          <li className="flex flex-col gap-1">
                            <code className="bg-base-200 px-3 py-2 rounded-lg font-mono">canvas.fill_background(r, g, b)</code>
                            <span className="text-sm text-gray-600 ml-4">Fill background color</span>
                          </li>
                          <li className="flex flex-col gap-1">
                            <code className="bg-base-200 px-3 py-2 rounded-lg font-mono">canvas.draw_polygon(points, fill=False)</code>
                            <span className="text-sm text-gray-600 ml-4">Draw polygon from points</span>
                          </li>
                          <li className="flex flex-col gap-1">
                            <code className="bg-base-200 px-3 py-2 rounded-lg font-mono">canvas.draw_text(x, y, text, font_size=16)</code>
                            <span className="text-sm text-gray-600 ml-4">Draw text</span>
                          </li>
                        </ul>
                        <div className="mt-8">
                          <h5 className="font-bold mb-4">Example Code</h5>
                          <pre className="bg-base-200 p-4 rounded-lg overflow-x-auto">
                            <code className="text-sm">{`# ArtCanvas Examples
from artcanvas import ArtCanvas
import math

def draw():
    with ArtCanvas() as canvas:
        # Fill background white
        canvas.fill_background(1, 1, 1)
        
        # Set color (r,g,b,a) - red with 50% transparency
        canvas.set_color(1, 0, 0, 0.5)
        
        # Set line width
        canvas.set_line_width(3)
        
        # Draw shapes
        canvas.draw_circle(200, 200, 50)  # Empty circle
        canvas.draw_circle(400, 200, 50, fill=True)  # Filled circle
        
        # Draw polygon
        points = [(800, 150), (900, 150), (850, 250)]
        canvas.draw_polygon(points, fill=True)
        
        # Draw text
        canvas.set_color(0, 0, 0)  # Black color
        canvas.draw_text(100, 600, "Hello World!", font_size=24)

draw()`}</code>
                          </pre>
                        </div>
                        <div className="mt-6 p-4 bg-base-200 rounded-lg">
                          <p className="text-sm text-gray-600">💡 All coordinates are relative to a 1920x1200 canvas</p>
                        </div>
                      </div>
      </div>
    </div>
  );
}; 