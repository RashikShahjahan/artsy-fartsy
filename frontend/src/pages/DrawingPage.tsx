import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalytics } from '../analytics';
import { editArtCode, retrieveArtCode, runArtCode, storeCode, type RunArtResult } from '../api';
import { Alert } from '../components/Alert';
import { CodeEditor } from '../components/CodeEditor';
import { DocumentationModal } from '../components/DocumentationModal';
import { DrawingCanvas } from '../components/DrawingCanvas';
import { Header } from '../components/Header';
import { PromptInput } from '../components/PromptInput';

function DrawingPage() {
  const [code, setCode] = useState('');
  const [image, setImage] = useState('');
  const [renderedCode, setRenderedCode] = useState('');
  const [executionToken, setExecutionToken] = useState('');
  const [currentPrompt, setCurrentPrompt] = useState('');
  const [editMode, setEditMode] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [showDocs, setShowDocs] = useState(false);
  const [initialPrompt, setInitialPrompt] = useState('');
  const [alert, setAlert] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const operationId = useRef(0);
  const navigate = useNavigate();
  const { trackEvent } = useAnalytics();

  useEffect(() => {
    return () => {
      operationId.current += 1;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (image) URL.revokeObjectURL(image);
    };
  }, [image]);

  const clearRenderedResult = () => {
    setImage('');
    setRenderedCode('');
    setExecutionToken('');
  };

  const acceptRenderedResult = (result: RunArtResult, executedCode: string, requestId: number): boolean => {
    if (operationId.current !== requestId) {
      URL.revokeObjectURL(result.imageUrl);
      return false;
    }
    setImage(result.imageUrl);
    setRenderedCode(executedCode);
    setExecutionToken(result.executionToken);
    return true;
  };

  const generateCode = async () => {
    const requestId = ++operationId.current;
    const prompt = currentPrompt.trim();
    setIsGenerating(true);
    trackEvent('generate_code', { prompt });

    try {
      const newCode = await retrieveArtCode(prompt, 'drawing');
      if (operationId.current !== requestId) return;

      setCode(newCode);
      setInitialPrompt(prompt);
      setEditMode(true);
      clearRenderedResult();

      try {
        setIsRunning(true);
        trackEvent('auto_run_code', { prompt });
        const result = await runArtCode(newCode, 'drawing');
        if (acceptRenderedResult(result, newCode, requestId)) {
          trackEvent('code_execution_success', { prompt });
        }
      } catch (error) {
        if (operationId.current !== requestId) return;
        const message = `Failed to run generated code: ${error instanceof Error ? error.message : 'Unknown error'}`;
        trackEvent('code_execution_error', { prompt, error: message });
        setAlert({ message, type: 'error' });
      } finally {
        if (operationId.current === requestId) setIsRunning(false);
      }
    } catch (error) {
      if (operationId.current !== requestId) return;
      const message = `Failed to generate code: ${error instanceof Error ? error.message : 'Unknown error'}`;
      trackEvent('code_generation_error', { prompt, error: message });
      setAlert({ message, type: 'error' });
    } finally {
      if (operationId.current === requestId) setIsGenerating(false);
    }
  };

  const runCode = async () => {
    const requestId = ++operationId.current;
    const codeToRun = code;
    clearRenderedResult();
    setIsRunning(true);
    trackEvent('manual_run_code', { code_length: codeToRun.length });

    try {
      const result = await runArtCode(codeToRun, 'drawing');
      if (acceptRenderedResult(result, codeToRun, requestId)) {
        trackEvent('manual_code_execution_success', { code_length: codeToRun.length });
      }
    } catch (error) {
      if (operationId.current !== requestId) return;
      const message = `Failed to run code: ${error instanceof Error ? error.message : 'Unknown error'}`;
      trackEvent('manual_code_execution_error', { error: message, code_length: codeToRun.length });
      setAlert({ message, type: 'error' });
    } finally {
      if (operationId.current === requestId) setIsRunning(false);
    }
  };

  const saveDrawing = async () => {
    if (!executionToken || !renderedCode || renderedCode !== code) {
      setAlert({ message: 'Run the current code successfully before saving', type: 'error' });
      return;
    }

    const codeToSave = renderedCode;
    const tokenToSave = executionToken;
    const promptToSave = initialPrompt;
    setIsSaving(true);
    trackEvent('save_drawing', { prompt: promptToSave });

    try {
      const success = await storeCode(promptToSave, codeToSave, 'drawing', tokenToSave);
      if (!success) throw new Error('Server returned an unsuccessful status code');
      trackEvent('save_drawing_success', { prompt: promptToSave });
      setAlert({ message: 'Drawing saved successfully', type: 'success' });
    } catch (error) {
      const message = `Failed to save drawing: ${error instanceof Error ? error.message : 'Unknown error'}`;
      trackEvent('save_drawing_error', { prompt: promptToSave, error: message });
      setAlert({ message, type: 'error' });
    } finally {
      setIsSaving(false);
    }
  };

  const editCode = async () => {
    const requestId = ++operationId.current;
    const prompt = currentPrompt.trim();
    const originalCode = code;
    setIsEditing(true);
    trackEvent('edit_code', { prompt, original_code_length: originalCode.length });

    try {
      const newCode = await editArtCode(prompt, originalCode, 'drawing');
      if (operationId.current !== requestId) return;

      setCode(newCode);
      clearRenderedResult();

      try {
        setIsRunning(true);
        trackEvent('run_edited_code', { prompt });
        const result = await runArtCode(newCode, 'drawing');
        if (acceptRenderedResult(result, newCode, requestId)) {
          trackEvent('edited_code_execution_success', { prompt });
        }
      } catch (error) {
        if (operationId.current !== requestId) return;
        const message = `Failed to run edited code: ${error instanceof Error ? error.message : 'Unknown error'}`;
        trackEvent('edited_code_execution_error', { prompt, error: message });
        setAlert({ message, type: 'error' });
      } finally {
        if (operationId.current === requestId) setIsRunning(false);
      }
    } catch (error) {
      if (operationId.current !== requestId) return;
      const message = `Failed to edit code: ${error instanceof Error ? error.message : 'Unknown error'}`;
      trackEvent('code_edit_error', { prompt, error: message });
      setAlert({ message, type: 'error' });
    } finally {
      if (operationId.current === requestId) setIsEditing(false);
    }
  };

  const handleCodeChange = (newCode: string) => {
    if (code && newCode !== code) {
      trackEvent('manual_code_edit', { changed_length: Math.abs(newCode.length - code.length) });
    }
    operationId.current += 1;
    setCode(newCode);
    clearRenderedResult();
  };

  const handleReset = () => {
    operationId.current += 1;
    trackEvent('reset_drawing');
    setInitialPrompt('');
    setCurrentPrompt('');
    setCode('');
    clearRenderedResult();
    setEditMode(false);
  };

  const promptIsLoading = isGenerating || isEditing || isRunning || isSaving;
  const canSave = Boolean(
    image
    && executionToken
    && renderedCode
    && renderedCode === code
    && initialPrompt.trim()
    && !promptIsLoading,
  );

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
        onSubmit={editMode ? editCode : generateCode}
        isLoading={promptIsLoading}
        placeholder={editMode ? 'Type your message...' : "E.g., 'A sunset over mountains'"}
        helperText={editMode ? 'Ask to make edits to your drawing' : 'Be specific with your description for better results'}
        submitButtonText={editMode ? 'Edit' : 'Generate Code'}
        loadingText={editMode ? 'Editing...' : 'Generating...'}
        showResetButton={editMode}
        onReset={handleReset}
        submitButtonClass={editMode ? 'btn-info' : 'btn-success'}
      />

      <div className="flex flex-col md:flex-row items-start justify-center gap-8">
        <CodeEditor
          code={code}
          onCodeChange={handleCodeChange}
          isRunning={promptIsLoading}
          onRun={runCode}
          onToggleDocs={() => {
            trackEvent('toggle_documentation', { showing: !showDocs });
            setShowDocs(!showDocs);
          }}
        />
        <DrawingCanvas
          image={image}
          isRunning={isRunning}
          isSaving={isSaving}
          canSave={canSave}
          onSave={saveDrawing}
        />
        <DocumentationModal
          isOpen={showDocs}
          onClose={() => {
            trackEvent('close_documentation');
            setShowDocs(false);
          }}
        />
      </div>

      <button
        onClick={() => {
          trackEvent('navigate_to_search');
          navigate('/search');
        }}
        className="btn btn-secondary w-full max-w-xs mx-auto"
        disabled={promptIsLoading}
      >
        Search for similar drawings
      </button>
    </div>
  );
}

export default DrawingPage;
