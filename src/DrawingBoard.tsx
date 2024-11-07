import { useState} from 'react';
import { modifyDrawing, retrieveArtCode, runDrawingCode } from './api';
import { useAuth } from '@clerk/clerk-react';

function DrawingBoard() {
  const [code, setCode] = useState('');
  const [image, setImage] = useState('');
  const [prompt, setPrompt] = useState(''); 
  const [promptHistory, setPromptHistory] = useState<string[]>([]);
  const { getToken } = useAuth();

  const generateCode = async () => {
    const token = await getToken();
    if (!token) {
      throw new Error('No token found');
    }
    if (promptHistory.length == 0) {
      const newCode = await retrieveArtCode(prompt, token);  
      setPromptHistory([prompt]);
      setCode(newCode.code);
    } else {
      const newCode = await modifyDrawing(promptHistory[promptHistory.length - 1], prompt, token);
      setPromptHistory([...promptHistory, prompt]);
      setCode(newCode);
    }
  };

  const runCode = async () => {
    const token = await getToken();
    if (!token) {
      throw new Error('No token found');
    }
    const image = await runDrawingCode(code, token);  
    setImage(image);
  };

  const shareDrawing = async () => {
    // TODO: Generate shareable link 
  };

  return (
    <div className="flex flex-col gap-4"> 
      <div className="flex gap-4 w-full max-w-4xl mx-auto">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Ask AI to draw..."
          className="flex-grow px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button 
          onClick={() => generateCode()}
          className="px-6 py-2 bg-green-400 text-black rounded-lg hover:bg-green-300 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-green-200 focus:ring-opacity-50"
        >
          Generate Code
        </button>
      </div>

      <div className="flex flex-col md:flex-row items-start justify-center gap-8">
        <div className="w-full md:w-1/2 h-[calc(100vh-16rem)] bg-white rounded-lg shadow-md p-6 flex flex-col relative">
          <img 
            src={image}
            alt="Drawing Board"
            className="w-full h-full object-contain mb-16" 
          />
          <button 
            onClick={() => shareDrawing()}
            className="absolute bottom-6 left-6 right-6 px-6 py-3 bg-yellow-400 text-black rounded-lg hover:bg-yellow-300 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-yellow-200 focus:ring-opacity-50"
          >
            Share
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
      </div>
    </div>
  );
}

export default DrawingBoard;
