import { useState} from 'react';
import { editArtCode, findSimilarArt, retrieveArtCode, runArtCode, storeCode } from './api';
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
  const [editMode, setEditMode] = useState(false);
  const [similarDrawings, setSimilarDrawings] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isFinding, setIsFinding] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [showDocs, setShowDocs] = useState(false);
  const [alert, setAlert] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

  const generateCode = async () => {
    try {
      setIsGenerating(true);
      const newCode = await retrieveArtCode(prompt, 'drawing');  
      setCode(newCode);
      setImage('');
      
      try {
        setIsRunning(true);
        const image = await runArtCode(newCode, 'drawing');
        setImage(image);
      } catch (error: any) {
        const message = error.type === 'malicious_code'
          ? 'This code contains potentially unsafe operations and cannot be executed'
          : 'Failed to run generated code: ' + (error instanceof Error ? error.message : 'Unknown error');
        
        setAlert({
          message,
          type: 'error'
        });
      } finally {
        setIsRunning(false);
      }
    } catch (error: any) {
      setAlert({
        message: 'Failed to generate code: ' + (error instanceof Error ? error.message : 'Unknown error'),
        type: 'error'
      });
    } finally {
      setIsGenerating(false);
      setEditMode(true);
    }
  };

  const runCode = async () => {
    try {
      setIsRunning(true);
      const image = await runArtCode(code, 'drawing');  
      setImage(image);
    } catch (error: any) {
      const message =  'Failed to run code: ' + (error instanceof Error ? error.message : 'Unknown error');
      
      setAlert({
        message,
        type: 'error'
      });
    } finally {
      setIsRunning(false);
    }
  };

  const findSimilar = async () => {
    try {
      setIsFinding(true);
      const response = await findSimilarArt(prompt, 'drawing');
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
      const success = await storeCode(prompt, code, 'drawing');
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

  const editCode = async () => {
    try {
      setIsEditing(true);
      const newCode = await editArtCode(prompt, code, 'drawing');  
      setCode(newCode);
      setImage('');
      
      try {
        setIsRunning(true);
        const image = await runArtCode(newCode, 'drawing');
        setImage(image);
      } catch (error: any) {
        const message = error.type === 'malicious_code'
          ? 'This code contains potentially unsafe operations and cannot be executed'
          : 'Failed to run generated code: ' + (error instanceof Error ? error.message : 'Unknown error');
        
        setAlert({
          message,
          type: 'error'
        });
      } finally {
        setIsRunning(false);
      }
    } catch (error: any) {
      setAlert({
        message: 'Failed to generate code: ' + (error instanceof Error ? error.message : 'Unknown error'),
        type: 'error'
      });
    } finally {
      setIsEditing(false);
    }
  }

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
        onSubmit={() => {
          if (editMode) {
            // Continue the conversation
            editCode();
          } else if (drawMode) {
            // Initial drawing generation
            generateCode();
          } else {
            // Search mode
            findSimilar();
          }
        }}
        isLoading={isGenerating || isFinding || isEditing}
        drawMode={drawMode}
        editMode={editMode}
        onReset={() => {
          setCode('');
          setImage('');
          setEditMode(false);
        }}
      />

      <div className="flex flex-col md:flex-row items-start justify-center gap-8">
        {drawMode ? (
          <>

            <CodeEditor 
              code={code}
              onCodeChange={setCode}
              isRunning={isRunning}
              onRun={runCode}
              onToggleDocs={() => setShowDocs(!showDocs)}
            />
                        <DrawingCanvas 
              image={image}
              isRunning={isRunning}
              isSaving={isSaving}
              onSave={saveDrawing}
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
