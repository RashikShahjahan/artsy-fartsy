import { useState} from 'react';
import { retrieveArtCode, runDrawingCode, storeCode } from './api';


function App() {
  const [code, setCode] = useState('');
  const [image, setImage] = useState('');
  const [prompt, setPrompt] = useState(''); 
  const [drawMode, setDrawMode] = useState(true);

  const generateCode = async () => {
    const newCode = await retrieveArtCode(prompt);  
    setCode(newCode.code);
    setImage('');
  };

  const runCode = async () => {
    const image = await runDrawingCode(code);  
    setImage(image);
  };

  const findSimilar = async () => {

  };

  const saveDrawing = async () => {
    await storeCode(code);
  };

  return (
    <div className="flex flex-col gap-4"> 
      <div className="flex gap-4 w-full max-w-4xl mx-auto">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={drawMode ? "Ask AI to draw..." : "Ask AI to find similar..."}
          className="flex-grow px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button 
          onClick={() => {drawMode ? generateCode() : findSimilar()}}
          className="px-6 py-2 bg-green-400 text-black rounded-lg hover:bg-green-300 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-green-200 focus:ring-opacity-50"
        >
          {drawMode ? "Draw" : "Find"}
        </button>
      </div>

      <div className="flex flex-col md:flex-row items-start justify-center gap-8">
        {drawMode ? (
          // Original drawing board and code editor layout
          <>
            <div className="w-full md:w-1/2 h-[calc(100vh-16rem)] bg-white rounded-lg shadow-md p-6 flex flex-col relative">
              <img 
                src={image}
                alt="Drawing Board"
                className="w-full h-full object-contain mb-16" 
              />
              <button 
                onClick={() => saveDrawing()}
                className="absolute bottom-6 left-6 right-6 px-6 py-3 bg-yellow-400 text-black rounded-lg hover:bg-yellow-300 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-yellow-200 focus:ring-opacity-50"
              >
                Save
              </button>
            </div>
            <div className="w-full md:w-1/2 h-[calc(100vh-16rem)] bg-white rounded-lg shadow-md p-6 flex flex-col">
              <textarea 
                value={code} 
                onChange={(e) => setCode(e.target.value)} 
                placeholder="Generated Code"
                className="w-full flex-grow border border-gray-300 rounded-lg resize-none p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <div className="flex gap-4 mt-6">
                <button 
                  onClick={() => runCode()}
                  className="w-full px-6 py-3 text-black rounded-lg transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-opacity-50 bg-blue-400 hover:bg-blue-300 focus:ring-blue-200"
                >
                  Run Code
                </button>
              </div>
            </div>
          </>
        ) : (
          // Three side-by-side images layout
          <>
            {[1, 2, 3].map((index) => (
              <div key={index} className="w-full md:w-1/3 h-[calc(100vh-16rem)] bg-white rounded-lg shadow-md p-6">
                <img 
                  src={image}
                  alt={`Similar Image ${index}`}
                  className="w-full h-full object-contain" 
                />
              </div>
            ))}
          </>
        )}
      </div>
      <button 
              onClick={() => {setDrawMode(!drawMode)}}
        className="w-1/2 mx-auto px-6 py-3 text-black rounded-lg transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-opacity-50 bg-purple-400 hover:bg-purple-300 focus:ring-purple-200"
        >
        {drawMode ? "Find Similar" : "Back to Drawing"}
      </button>
    </div>
  );
}

export default App;
