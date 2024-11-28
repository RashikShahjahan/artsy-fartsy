import { useState} from 'react';
import { editArtCode, retrieveArtCode, runArtCode, storeCode } from '../api';
import { Header } from '../components/Header';
import { PromptInput } from '../components/PromptInput';
import { DrawingCanvas } from '../components/DrawingCanvas';
import { CodeEditor } from '../components/CodeEditor';
import { DocumentationModal } from '../components/DocumentationModal';
import { Alert } from '../components/Alert';
import { useNavigate } from 'react-router-dom';

function DrawingPage() {
  const [code, setCode] = useState('');
  const [image, setImage] = useState('');
  const [currentPrompt, setCurrentPrompt] = useState(''); 
  const [editMode, setEditMode] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [showDocs, setShowDocs] = useState(false);
  const [initialPrompt, setInitialPrompt] = useState('');
  const [alert, setAlert] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const navigate = useNavigate();

  const generateCode = async () => {
    try {
      setIsGenerating(true);
      setInitialPrompt(currentPrompt);
      const newCode = await retrieveArtCode(currentPrompt, 'drawing');  
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

  const saveDrawing = async () => {
    try {
      setIsSaving(true);
      const success = await storeCode(initialPrompt, code, 'drawing');
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
      const newCode = await editArtCode(currentPrompt, code, 'drawing');  
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
      
      <Header drawMode={true} />
      
      <PromptInput 
        prompt={currentPrompt}
        onPromptChange={setCurrentPrompt}
        onSubmit={() => {
          if (editMode) {
            editCode();
          } else {
            generateCode();
          }
        }}
        isLoading={isGenerating || isEditing}
        placeholder={
          editMode 
            ? "Type your message..."
            : "E.g., 'A sunset over mountains'"
        }
        helperText={
          editMode
            ? "Ask to make edits to your drawing"
            : "Be specific with your description for better results"
        }
        submitButtonText={editMode ? "Edit" : "Generate Code"}
        loadingText={editMode ? "Editing..." : "Generating..."}
        showResetButton={editMode}
        onReset={() => {
          setInitialPrompt('');
          setCode('');
          setImage('');
          setEditMode(false);
        }}
        submitButtonClass={editMode ? 'btn-info' : 'btn-success'}
      />

      <div className="flex flex-col md:flex-row items-start justify-center gap-8">
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
      </div>

      <button   
        onClick={() => navigate('/search')}
        className="btn btn-secondary w-full max-w-xs mx-auto"
      >
        Search for similar drawings
      </button>
      
    </div>
  );
}

export default DrawingPage;
