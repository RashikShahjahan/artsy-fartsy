/*Saving Drawings and viewing gallery(Add auth)
  - On next or previous button click, pass back current artId to backend
  - Backend will return the next or previous artId
  - Update the artId state
  - Pass artid to ArtPiece component
  - ArtPiece component will fetch the drawcommands from backend using artid along with metadata such as username, likes, etc.
  - Artpiece component will render  the drawcommands on a canvas
  Post request to like/unlike drawings
*/

import { useState } from "react";
import ArtPiece from "./ArtPiece";

export default function Gallery() {
    const [artId, setArtId] = useState(0);

    return (
      <div>
        <ArtPiece artId={artId} />
        <button className="w-1/2 bottom-6 left-6 right-6 px-6 py-3 bg-yellow-400 text-black rounded-lg hover:bg-yellow-300 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-yellow-200 focus:ring-opacity-50">
            Previous
        </button>
        <button 
          className="w-1/2 bottom-6 left-6 right-6 px-6 py-3 bg-pink-400 text-black rounded-lg hover:bg-pink-300 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-pink-200 focus:ring-opacity-50 mt-6"
        >
          Next
        </button>
      </div>

    )
  }