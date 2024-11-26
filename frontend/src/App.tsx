import { useState} from 'react';
import { findSimilarDrawing, retrieveArtCode, runDrawingCode, storeCode } from './api';
import { Header } from './components/Header';
import { PromptInput } from './components/PromptInput';
import { DrawingCanvas } from './components/DrawingCanvas';
import { CodeEditor } from './components/CodeEditor';
import { DocumentationModal } from './components/DocumentationModal';
import { SimilarDrawings } from './components/SimilarDrawings';
import { Alert } from './components/Alert';

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
  const [showDocs, setShowDocs] = useState(false);
  const [alert, setAlert] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

  const generateCode = async () => {
    try {
      setIsGenerating(true);
      const newCode = await retrieveArtCode(prompt);  
      setCode(newCode.code);
      setImage('');
      const image = await runDrawingCode(newCode.code);
      setImage(image);
    } catch (error) {
      setAlert({
        message: 'Failed to generate code: ' + (error instanceof Error ? error.message : 'Unknown error'),
        type: 'error'
      });
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
      setAlert({
        message: 'Failed to run code: ' + (error instanceof Error ? error.message : 'Unknown error'),
        type: 'error'
      });
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
      setAlert({
        message: 'Failed to find similar drawings: ' + (error instanceof Error ? error.message : 'Unknown error'),
        type: 'error'
      });
    } finally {
      setIsFinding(false);
    }
  };

  const saveDrawing = async () => {
    try {
      setIsSaving(true);
      const success = await storeCode(code);
      if (success) {
        setAlert({
          message: 'Drawing saved successfully',
          type: 'success'
        });
      } else {
        throw new Error('Server returned an unsuccessful status code');
      }
    } catch (error) {
      setAlert({
        message: 'Failed to save drawing: ' + (error instanceof Error ? error.message : 'Unknown error'),
        type: 'error'
      });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 container mx-auto p-6">
      {alert && (
        <Alert
          message={alert.message}
          type={alert.type}
          onClose={() => setAlert(null)}
        />
      )}
      
      <Header drawMode={drawMode} />
      
      <PromptInput 
        prompt={prompt}
        onPromptChange={setPrompt}
        onSubmit={() => drawMode ? generateCode() : findSimilar()}
        isLoading={isGenerating || isFinding}
        drawMode={drawMode}
      />

      <div className="flex flex-col md:flex-row items-start justify-center gap-8">
        {drawMode ? (
          <>
            <DrawingCanvas 
              image={image}
              isRunning={isRunning}
              isSaving={isSaving}
              onSave={saveDrawing}
            />
            <CodeEditor 
              code={code}
              onCodeChange={setCode}
              isRunning={isRunning}
              onRun={runCode}
              onToggleDocs={() => setShowDocs(!showDocs)}
            />
            <DocumentationModal 
              isOpen={showDocs}
              onClose={() => setShowDocs(false)}
            />
          </>
        ) : (
          <SimilarDrawings 
            drawings={similarDrawings}
            isFinding={isFinding}
          />
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
