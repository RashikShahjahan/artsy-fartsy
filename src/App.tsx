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
    <div className="flex flex-col gap-6 container mx-auto p-6">
      <header className="text-center">
        <h1 className="text-3xl font-bold mb-2">AI Drawing Generator</h1>
        <p className="text-gray-600">
          {drawMode 
            ? "Describe what you want to draw, and let AI generate the code for you!" 
            : "Find similar drawings based on your description"}
        </p>
      </header>

      <div className="flex gap-4 w-full max-w-4xl mx-auto">
        <div className="form-control flex-grow">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={drawMode ? "E.g., 'A sunset over mountains'" : "E.g., 'Abstract art with circles'"}
            className="input input-bordered w-full"
            disabled={isGenerating}
          />
          <label className="label">
            <span className="label-text-alt text-gray-500">
              {drawMode ? "Be specific with your description for better results" : "Enter keywords to find similar artwork"}
            </span>
          </label>
        </div>
        <button 
          onClick={() => {drawMode ? generateCode() : findSimilar()}}
          className={`btn ${drawMode ? 'btn-success' : 'btn-primary'} min-w-[120px]`}
          disabled={isGenerating || isFinding || !prompt.trim()}
          title={!prompt.trim() ? "Please enter a description first" : ""}
        >
          {drawMode ? (
            isGenerating ? (
              <>
                <span className="loading loading-spinner"></span>
                Generating...
              </>
            ) : "Generate Code"
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
            <div className="card w-full md:w-1/2 h-[calc(100vh-20rem)] bg-base-100 shadow-xl">
              <div className="card-body relative">
                {!image && !isRunning && (
                  <div className="absolute inset-0 flex items-center justify-center text-center text-gray-500">
                    <div>
                      <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      <p>Your drawing will appear here</p>
                    </div>
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
            <div className="card w-full md:w-1/2 h-[calc(100vh-20rem)] bg-base-100 shadow-xl">
              <div className="card-body">
                <h3 className="text-lg font-semibold mb-2">Generated Code</h3>
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
          <div className="w-full grid grid-cols-1 md:grid-cols-3 gap-6">
            {similarDrawings.length === 0 && !isFinding && (
              <div className="col-span-full text-center text-gray-500 py-12">
                <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <p>Enter a description to find similar drawings</p>
              </div>
            )}
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
          </div>
        )}
      </div>

      <button 
        onClick={() => {setDrawMode(!drawMode)}}
        className="btn btn-secondary w-full max-w-xs mx-auto"
      >
        {drawMode ? "Search for similar drawings" : "Back to Drawing"}
      </button>
    </div>
  );
}

export default App;
