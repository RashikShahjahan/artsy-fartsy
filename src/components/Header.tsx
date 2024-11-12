interface HeaderProps {
  drawMode: boolean;
}

export const Header = ({ drawMode }: HeaderProps) => (
  <header className="text-center">
    <h1 className="text-3xl font-bold mb-2">AI Drawing Generator</h1>
    <p className="text-gray-600 mb-2">
      {drawMode 
        ? "Describe what you want to draw, and let AI generate Python code using our ArtCanvas library!" 
        : "Find similar drawings based on your description"}
    </p>
    {drawMode && (
      <p className="text-sm text-gray-500">
        Your code will use the <code className="bg-gray-100 px-1 rounded">ArtCanvas</code> library to draw. 
        Click "Show Docs" in the code editor to see available drawing commands.
      </p>
    )}
  </header>
); 