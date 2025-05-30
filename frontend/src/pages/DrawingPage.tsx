import { useState} from 'react';
import { editArtCode, retrieveArtCode, runArtCode, storeCode } from '../api';
import { Header } from '../components/Header';
import { PromptInput } from '../components/PromptInput';
import { DrawingCanvas } from '../components/DrawingCanvas';
import { CodeEditor } from '../components/CodeEditor';
import { DocumentationModal } from '../components/DocumentationModal';
import { Alert } from '../components/Alert';
import { useNavigate } from 'react-router-dom';
import { useAnalytics } from 'rashik-analytics-provider';
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
  const { trackEvent } = useAnalytics();

  const generateCode = async () => {
    try {
      setIsGenerating(true);
      setInitialPrompt(currentPrompt);
      trackEvent('generate_code', { prompt: currentPrompt });
      const newCode = await retrieveArtCode(currentPrompt, 'drawing');  
      setCode(newCode);
      setImage('');
      setEditMode(true);
      
      try {
        setIsRunning(true);
        trackEvent('auto_run_code', { prompt: currentPrompt });
        const image = await runArtCode(newCode, 'drawing', true);
        setImage(image);
        trackEvent('code_execution_success', { prompt: currentPrompt });
      } catch (error: any) {
        const message = error.type === 'malicious_code'
          ? 'This code contains potentially unsafe operations and cannot be executed'
          : 'Failed to run generated code: ' + (error instanceof Error ? error.message : 'Unknown error');
        
        trackEvent('code_execution_error', { 
          prompt: currentPrompt,
          error: message,
          error_type: error.type || 'unknown'
        });
        
        setAlert({
          message,
          type: 'error'
        });
      } finally {
        setIsRunning(false);
      }
    } catch (error: any) {
      trackEvent('code_generation_error', { 
        prompt: currentPrompt,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      
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
      trackEvent('manual_run_code', { code_length: code.length });
      const image = await runArtCode(code, 'drawing', false);  
      setImage(image);
      trackEvent('manual_code_execution_success', { code_length: code.length });
    } catch (error: any) {
      const message =  'Failed to run code: ' + (error instanceof Error ? error.message : 'Unknown error');
      
      trackEvent('manual_code_execution_error', { 
        error: message,
        code_length: code.length
      });
      
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
      trackEvent('save_drawing', { prompt: initialPrompt });
      const success = await storeCode(initialPrompt, code, 'drawing');
      if (success) {
        trackEvent('save_drawing_success', { prompt: initialPrompt });
        setAlert({
          message: 'Drawing saved successfully',
          type: 'success'
        });
      } else {
        throw new Error('Server returned an unsuccessful status code');
      }
    } catch (error) {
      trackEvent('save_drawing_error', { 
        prompt: initialPrompt,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      
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
      trackEvent('edit_code', { prompt: currentPrompt, original_code_length: code.length });
      const newCode = await editArtCode(currentPrompt, code, 'drawing');  
      setCode(newCode);
      setImage('');
      
      try {
        setIsRunning(true);
        trackEvent('run_edited_code', { prompt: currentPrompt });
        const image = await runArtCode(newCode, 'drawing', true);
        setImage(image);
        trackEvent('edited_code_execution_success', { prompt: currentPrompt });
      } catch (error: any) {
        const message = error.type === 'malicious_code'
          ? 'This code contains potentially unsafe operations and cannot be executed'
          : 'Failed to run generated code: ' + (error instanceof Error ? error.message : 'Unknown error');
        
        trackEvent('edited_code_execution_error', { 
          prompt: currentPrompt,
          error: message,
          error_type: error.type || 'unknown'
        });
        
        setAlert({
          message,
          type: 'error'
        });
      } finally {
        setIsRunning(false);
      }
    } catch (error: any) {
      trackEvent('code_edit_error', { 
        prompt: currentPrompt,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      
      setAlert({
        message: 'Failed to generate code: ' + (error instanceof Error ? error.message : 'Unknown error'),
        type: 'error'
      });
    } finally {
      setIsEditing(false);
    }
  }

  const navigateToSearch = () => {
    trackEvent('navigate_to_search');
    navigate('/search');
  };

  const handleCodeChange = (newCode: string) => {
    if (code && newCode !== code) {
      trackEvent('manual_code_edit', { 
        changed_length: Math.abs(newCode.length - code.length)
      });
    }
    setCode(newCode);
  };

  const handleToggleDocs = () => {
    trackEvent('toggle_documentation', { showing: !showDocs });
    setShowDocs(!showDocs);
  };

  const handleReset = () => {
    trackEvent('reset_drawing');
    setInitialPrompt('');
    setCode('');
    setImage('');
    setEditMode(false);
  };

  return (
    <div className="flex flex-col gap-6 container mx-auto p-6">
      {alert && (
        <Alert
          message={alert.message}
          type={alert.type}
          onClose={() => {
            trackEvent('close_alert', { alert_type: alert.type });
            setAlert(null);
          }}
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
        onReset={handleReset}
        submitButtonClass={editMode ? 'btn-info' : 'btn-success'}
      />

      <div className="flex flex-col md:flex-row items-start justify-center gap-8">
        <>
            <CodeEditor 
              code={code}
              onCodeChange={handleCodeChange}
              isRunning={isRunning}
              onRun={runCode}
              onToggleDocs={handleToggleDocs}
            />
            <DrawingCanvas 
              image={image}
              isRunning={isRunning}
              isSaving={isSaving}
              onSave={saveDrawing}
            />
            <DocumentationModal 
              isOpen={showDocs}
              onClose={() => {
                trackEvent('close_documentation');
                setShowDocs(false);
              }}
            />
        </>
      </div>

      <button   
        onClick={navigateToSearch}
        className="btn btn-secondary w-full max-w-xs mx-auto"
      >
        Search for similar drawings
      </button>
      
    </div>
  );
}

export default DrawingPage;
