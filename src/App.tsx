import { SignedIn, SignedOut, SignIn } from "@clerk/clerk-react";
import { useState } from "react";
import DrawingBoard from "./DrawingBoard";
import Gallery from "./Gallery";

export default function App() {
  const [drawMode, setDrawMode] = useState(true);

  return (
    <div className="flex flex-col items-center min-h-screen bg-gray-100 p-4 sm:p-6 md:p-8">
      <div className="w-full max-w-7xl">
        <button
          onClick={() => setDrawMode(!drawMode)}
          className="w-full mb-8 py-4 px-6 text-xl font-bold text-black bg-red-400 rounded-lg shadow-md hover:bg-red-300 focus:outline-none focus:ring-2 focus:ring-red-300 focus:ring-opacity-50 transition-colors duration-300"
        >
          Switch to {drawMode ? 'Gallery' : 'Drawing Board'}
        </button>
        
        <SignedIn>
          {drawMode ? <DrawingBoard auth={true} /> : <Gallery />}
        </SignedIn>
        <SignedOut>
          {drawMode ? <DrawingBoard auth={false} /> : <SignIn />}
        </SignedOut>
      </div>
    </div>
  )
}
