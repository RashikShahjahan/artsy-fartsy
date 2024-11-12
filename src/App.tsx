import { useState} from 'react';
import { findSimilarDrawing, retrieveArtCode, runDrawingCode, storeCode } from './api';
import { Header } from './components/Header/Header';
import { PromptInput } from './components/PromptInput/PromptInput';
import { DrawingCanvas } from './components/DrawingCanvas/DrawingCanvas';
import { CodeEditor } from './components/CodeEditor/CodeEditor';
import { DocumentationModal } from './components/DocumentationModal/DocumentationModal';
import { SimilarDrawings } from './components/SimilarDrawings/SimilarDrawings';
import { Button } from './components/common/Button/Button';

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

      <Button 
        variant="secondary"
        onClick={() => setDrawMode(!drawMode)}
        className="max-w-xs mx-auto"
      >
        {drawMode ? "Search for similar drawings" : "Back to Drawing"}
      </Button>
    </div>
  );
}

export default App;
