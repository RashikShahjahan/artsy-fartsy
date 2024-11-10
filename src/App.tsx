import { useState} from 'react';
import { findSimilarDrawing, retrieveArtCode, runDrawingCode, storeCode } from './api';

function App() {
  const [code, setCode] = useState('');
  const [image, setImage] = useState('');
  const [prompt, setPrompt] = useState(''); 
  const [drawMode, setDrawMode] = useState(true);
  const [similarDrawings, setSimilarDrawings] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isFinding, setIsFinding] = useState(false);

  const generateCode = async () => {
    try {
      setIsGenerating(true);
      const newCode = await retrieveArtCode(prompt);  
      setCode(newCode.code);
      setImage('');
    } catch (error) {
      alert('Failed to generate code: ' + (error instanceof Error ? error.message : 'Unknown error'));
    } finally {
      setIsGenerating(false);
    }
  };

  const runCode = async () => {
    try {
      setIsRunning(true);
      const image = await runDrawingCode(code);  
      setImage(image);
    } catch (error) {
      alert('Failed to run code: ' + (error instanceof Error ? error.message : 'Unknown error'));
    } finally {
      setIsRunning(false);
    }
  };

  const findSimilar = async () => {
    try {
      setIsFinding(true);
      const response = await findSimilarDrawing(prompt);
      setSimilarDrawings(response);
    } catch (error) {
      alert('Failed to find similar drawings: ' + (error instanceof Error ? error.message : 'Unknown error'));
    } finally {
      setIsFinding(false);
    }
  };

  const saveDrawing = async () => {
    try {
      setIsSaving(true);
      await storeCode(code);
    } catch (error) {
      alert('Failed to save drawing: ' + (error instanceof Error ? error.message : 'Unknown error'));
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 container mx-auto p-4"> 
      <div className="flex gap-4 w-full max-w-4xl mx-auto">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={drawMode ? "Ask AI to draw..." : "Ask AI to find similar..."}
          className="input input-bordered flex-grow"
          disabled={isGenerating}
        />
        <button 
          onClick={() => {drawMode ? generateCode() : findSimilar()}}
          className="btn btn-success"
          disabled={isGenerating || isFinding}
        >
          {drawMode ? (
            isGenerating ? (
              <>
                <span className="loading loading-spinner"></span>
                Drawing...
              </>
            ) : "Draw"
          ) : (
            isFinding ? (
              <>
                <span className="loading loading-spinner"></span>
                Finding...
              </>
            ) : "Find"
          )}
        </button>
      </div>

      <div className="flex flex-col md:flex-row items-start justify-center gap-8">
        {drawMode ? (
          <>
            <div className="card w-full md:w-1/2 h-[calc(100vh-16rem)] bg-base-100 shadow-xl">
              <div className="card-body relative">
                {isRunning && (
                  <div className="absolute inset-0 flex items-center justify-center bg-base-100/50">
                    <span className="loading loading-spinner loading-lg"></span>
                  </div>
                )}
                <img 
                  src={image}
                  alt="Drawing Board"
                  className="w-full h-full object-contain mb-16" 
                />
                <button 
                  onClick={() => saveDrawing()}
                  className="btn btn-warning absolute bottom-6 left-6 right-6"
                  disabled={isSaving}
                >
                  {isSaving ? (
                    <>
                      <span className="loading loading-spinner"></span>
                      Saving...
                    </>
                  ) : "Save"}
                </button>
              </div>
            </div>
            <div className="card w-full md:w-1/2 h-[calc(100vh-16rem)] bg-base-100 shadow-xl">
              <div className="card-body">
                <textarea 
                  value={code} 
                  onChange={(e) => setCode(e.target.value)} 
                  placeholder="Generated Code"
                  className="textarea textarea-bordered h-full"
                  disabled={isRunning}
                />
                <div className="card-actions justify-end mt-6">
                  <button 
                    onClick={() => runCode()}
                    className="btn btn-primary w-full"
                    disabled={isRunning}
                  >
                    {isRunning ? (
                      <>
                        <span className="loading loading-spinner"></span>
                        Running...
                      </>
                    ) : "Run Code"}
                  </button>
                </div>
              </div>
            </div>
          </>
        ) : (
          <>
            {similarDrawings.map((drawing, index) => (
              <div key={index} className="card w-full md:w-1/3 h-[calc(100vh-16rem)] bg-base-100 shadow-xl">
                <div className="card-body">
                  <img 
                    src={drawing}
                    alt={`Similar Image ${index}`}
                    className="w-full h-full object-contain" 
                  />
                </div>
              </div>
            ))}
          </>
        )}
      </div>
      <button 
        onClick={() => {setDrawMode(!drawMode)}}
        className="btn btn-secondary w-1/2 mx-auto"
      >
        {drawMode ? "Search for similar" : "Back to Drawing"}
      </button>
    </div>
  );
}

export default App;
