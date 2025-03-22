import React, { useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { javascript } from '@codemirror/lang-javascript';
import { java } from '@codemirror/lang-java';
import { cpp } from '@codemirror/lang-cpp';
import { php } from '@codemirror/lang-php';
import { rust } from '@codemirror/lang-rust';
import { vscodeDark } from '@uiw/codemirror-theme-vscode';
import { Code2, Wand2, FileCode, Download, Upload } from 'lucide-react';
import toast from 'react-hot-toast';

type Language = 'python' | 'javascript' | 'java' | 'csharp' | 'ruby' | 'php' | 'go' | 'rust';

interface ModernizationResult {
  success: boolean;
  modernized_code: string;
  changes_made: string[];
  language: string;
  frameworks_detected: string[];
  error_message?: string;
}

function App() {
  const [code, setCode] = useState('');
  const [modernizedCode, setModernizedCode] = useState('');
  const [language, setLanguage] = useState<Language>('python');
  const [loading, setLoading] = useState(false);
  const [detectedFrameworks, setDetectedFrameworks] = useState<string[]>([]);

  const getLanguageExtension = (lang: Language) => {
    switch (lang) {
      case 'python': return python();
      case 'javascript': return javascript();
      case 'java': return java();
      case 'csharp': return cpp();
      case 'php': return php();
      case 'rust': return rust();
      default: return python();
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      setCode(content);
      
      const extension = file.name.split('.').pop()?.toLowerCase();
      const languageMap: Record<string, Language> = {
        'py': 'python',
        'js': 'javascript',
        'java': 'java',
        'cs': 'csharp',
        'php': 'php',
        'rs': 'rust',
        'rb': 'ruby',
        'go': 'go'
      };
      
      if (extension && languageMap[extension]) {
        setLanguage(languageMap[extension]);
      }
    };
    reader.readAsText(file);
  };

  const handleDownload = (type: 'original' | 'modernized') => {
    const content = type === 'original' ? code : modernizedCode;
    if (!content) {
      toast.error(`No ${type} code to download`);
      return;
    }

    const extensions: Record<Language, string> = {
      python: 'py',
      javascript: 'js',
      java: 'java',
      csharp: 'cs',
      ruby: 'rb',
      php: 'php',
      go: 'go',
      rust: 'rs'
    };

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${type}_code.${extensions[language]}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleModernize = async () => {
    if (!code.trim()) {
      toast.error('Please enter some code to modernize');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/modernize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code,
          language,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to modernize code');
      }

      const data: ModernizationResult = await response.json();
      setModernizedCode(data.modernized_code);
      setDetectedFrameworks(data.frameworks_detected);
      
      if (data.changes_made?.length > 0) {
        toast.success(
          <div>
            <strong>Changes made:</strong>
            <ul className="mt-2 list-disc pl-4">
              {data.changes_made.map((change: string, index: number) => (
                <li key={index} className="text-sm">{change}</li>
              ))}
            </ul>
          </div>,
          { duration: 5000 }
        );
      }

      if (data.frameworks_detected.length > 0) {
        toast.success(
          <div>
            <strong>Frameworks detected:</strong>
            <ul className="mt-2 list-disc pl-4">
              {data.frameworks_detected.map((framework: string, index: number) => (
                <li key={index} className="text-sm">{framework}</li>
              ))}
            </ul>
          </div>,
          { duration: 5000 }
        );
      }
    } catch (error) {
      toast.error('Failed to modernize code. Please try again.');
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-stackblitz text-stackblitz p-8">
      <div className="max-w-7xl mx-auto animate-fade-in">
        <div className="flex items-center justify-between mb-8 animate-slide-in">
          <div className="flex items-center gap-3">
            <Code2 className="w-8 h-8 text-blue-400" />
            <h1 className="text-3xl font-bold">AI Code Modernizer</h1>
          </div>
          <div className="flex items-center gap-4">
            <label className="relative cursor-pointer">
              <input
                type="file"
                className="hidden"
                onChange={handleFileUpload}
                accept=".py,.js,.java,.cs,.rb,.php,.go,.rs"
              />
              <div className="flex items-center gap-2 px-4 py-2 bg-stackblitz-light border border-gray-700 rounded-lg hover:bg-opacity-80 transition-all duration-200 hover:border-blue-500">
                <Upload className="w-5 h-5 text-blue-400" />
                <span className="text-sm">Upload File</span>
              </div>
            </label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as Language)}
              className="px-4 py-2 rounded-lg bg-stackblitz-light border border-gray-700 text-stackblitz focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200"
            >
              <option value="python">Python</option>
              <option value="javascript">JavaScript</option>
              <option value="java">Java</option>
              <option value="csharp">C#</option>
              <option value="ruby">Ruby</option>
              <option value="php">PHP</option>
              <option value="go">Go</option>
              <option value="rust">Rust</option>
            </select>
            <button
              onClick={handleModernize}
              disabled={loading}
              className={`flex items-center gap-2 px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 ${loading ? 'animate-glow' : ''}`}
            >
              <Wand2 className="w-5 h-5" />
              {loading ? 'Modernizing...' : 'Modernize'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-8 animate-slide-in" style={{ animationDelay: '0.1s' }}>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Original Code</h2>
              <button
                onClick={() => handleDownload('original')}
                className="flex items-center gap-2 px-3 py-1 text-sm text-blue-400 hover:text-blue-300 transition-colors duration-200"
              >
                <Download className="w-4 h-4" />
                Download
              </button>
            </div>
            <div className="h-[600px] rounded-lg overflow-hidden border border-gray-700 shadow-lg transition-all duration-200 hover:border-blue-500">
              <CodeMirror
                value={code}
                height="600px"
                theme={vscodeDark}
                extensions={[getLanguageExtension(language)]}
                onChange={(value) => setCode(value)}
                className="h-full"
              />
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Modernized Code</h2>
              <button
                onClick={() => handleDownload('modernized')}
                className="flex items-center gap-2 px-3 py-1 text-sm text-blue-400 hover:text-blue-300 transition-colors duration-200"
              >
                <Download className="w-4 h-4" />
                Download
              </button>
            </div>
            <div className="h-[600px] rounded-lg overflow-hidden border border-gray-700 shadow-lg transition-all duration-200 hover:border-blue-500">
              <CodeMirror
                value={modernizedCode}
                height="600px"
                theme={vscodeDark}
                extensions={[getLanguageExtension(language)]}
                editable={false}
                className="h-full"
              />
            </div>
          </div>
        </div>

        {detectedFrameworks.length > 0 && (
          <div className="mt-6 p-4 bg-stackblitz-light rounded-lg shadow-lg border border-gray-700 animate-slide-in" style={{ animationDelay: '0.2s' }}>
            <h3 className="text-lg font-semibold mb-2">Detected Frameworks</h3>
            <div className="flex gap-2 flex-wrap">
              {detectedFrameworks.map((framework, index) => (
                <span
                  key={index}
                  className="px-3 py-1 bg-blue-500 bg-opacity-20 text-blue-400 rounded-full text-sm border border-blue-500 transition-all duration-200 hover:bg-opacity-30"
                >
                  {framework}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;