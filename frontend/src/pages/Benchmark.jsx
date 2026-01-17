import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Play, Terminal as TerminalIcon, Settings, Cpu, FileJson, AlertCircle, CheckCircle2 } from 'lucide-react';

const API_BASE = '/api';

export default function Benchmark() {
  const [loading, setLoading] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState(null);
  const [logs, setLogs] = useState([]);
  
  // Form State
  const [config, setConfig] = useState({
    api_provider: 'gemini',
    api_base_url: '',
    api_key: '',
    model_name: 'gemini-1.5-pro',
    input_mode: 'video',
    num_frames: 8,
    system_prompt: '',
    temperature: 0.2,
    max_tokens: 1000,
    data_root_dir: 'data'
  });

  const logEndRef = useRef(null);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Polling
  useEffect(() => {
    let interval;
    if (taskId && taskStatus?.status !== 'completed' && taskStatus?.status !== 'failed') {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`${API_BASE}/tasks/${taskId}`);
          setTaskStatus(res.data);
          setLogs(res.data.logs || []);
          
          if (res.data.status === 'completed' || res.data.status === 'failed') {
            setLoading(false);
            clearInterval(interval);
          }
        } catch (e) {
          console.error("Polling error", e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [taskId, taskStatus?.status]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setLogs([]);
    setTaskStatus(null);
    try {
      const res = await axios.post(`${API_BASE}/evaluate`, config);
      setTaskId(res.data.task_id);
      setTaskStatus({ status: 'pending', progress: 0 });
    } catch (err) {
      console.error(err);
      setLoading(false);
      alert("Failed to start evaluation: " + err.message);
    }
  };

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: type === 'number' ? Number(value) : value
    }));
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-12 grid lg:grid-cols-3 gap-8">
      {/* Left Column: Configuration */}
      <div className="lg:col-span-1 space-y-6">
        <div className="glass-panel p-6 rounded-lg">
          <div className="flex items-center gap-2 mb-6 text-white">
            <Settings className="w-5 h-5" />
            <h2 className="font-semibold text-lg">Configuration</h2>
          </div>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* API Settings */}
            <div className="space-y-3">
              <label className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Model Provider</label>
              <select 
                name="api_provider" 
                value={config.api_provider}
                onChange={handleChange}
                className="w-full bg-black border border-zinc-800 rounded px-3 py-2 text-sm text-white focus:border-white outline-none transition-colors"
              >
                <option value="gemini">Google Gemini</option>
                <option value="openai">OpenAI Compatible</option>
              </select>
            </div>

            <div className="space-y-3">
              <label className="text-xs font-bold text-zinc-500 uppercase tracking-wider">API Key</label>
              <input 
                type="password"
                name="api_key"
                value={config.api_key}
                onChange={handleChange}
                placeholder="sk-..."
                className="w-full bg-black border border-zinc-800 rounded px-3 py-2 text-sm text-white focus:border-white outline-none transition-colors"
                required
              />
            </div>

            {config.api_provider === 'openai' && (
              <div className="space-y-3">
                <label className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Base URL</label>
                <input 
                  type="text"
                  name="api_base_url"
                  value={config.api_base_url}
                  onChange={handleChange}
                  placeholder="https://api.openai.com/v1"
                  className="w-full bg-black border border-zinc-800 rounded px-3 py-2 text-sm text-white focus:border-white outline-none transition-colors"
                />
              </div>
            )}

            <div className="space-y-3">
              <label className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Model Name</label>
              <input 
                type="text"
                name="model_name"
                value={config.model_name}
                onChange={handleChange}
                className="w-full bg-black border border-zinc-800 rounded px-3 py-2 text-sm text-white focus:border-white outline-none transition-colors"
                required
              />
            </div>

            <div className="pt-4 border-t border-zinc-800 space-y-3">
               <label className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Input Mode</label>
               <div className="grid grid-cols-2 gap-2">
                 <button
                   type="button"
                   onClick={() => setConfig(c => ({...c, input_mode: 'video'}))}
                   className={`px-3 py-2 text-sm rounded border ${config.input_mode === 'video' ? 'bg-white text-black border-white' : 'bg-black text-zinc-400 border-zinc-800'}`}
                 >
                   Video File
                 </button>
                 <button
                   type="button"
                   onClick={() => setConfig(c => ({...c, input_mode: 'frames'}))}
                   className={`px-3 py-2 text-sm rounded border ${config.input_mode === 'frames' ? 'bg-white text-black border-white' : 'bg-black text-zinc-400 border-zinc-800'}`}
                 >
                   Frames
                 </button>
               </div>
            </div>

             {config.input_mode === 'frames' && (
                <div className="space-y-3">
                  <label className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Num Frames</label>
                  <input 
                    type="number"
                    name="num_frames"
                    value={config.num_frames}
                    onChange={handleChange}
                    className="w-full bg-black border border-zinc-800 rounded px-3 py-2 text-sm text-white focus:border-white outline-none transition-colors"
                  />
                </div>
             )}

            <div className="pt-4">
              <button 
                type="submit" 
                disabled={loading}
                className="w-full bg-white text-black font-bold py-3 rounded hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
              >
                {loading ? <Cpu className="animate-spin" size={18} /> : <Play size={18} />}
                {loading ? 'Evaluating...' : 'Start Evaluation'}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Right Column: Monitor & Results */}
      <div className="lg:col-span-2 space-y-6">
        {/* Status Panel */}
        <div className="glass-panel p-6 rounded-lg min-h-[400px] flex flex-col">
          <div className="flex items-center justify-between mb-4 text-white">
            <div className="flex items-center gap-2">
                <TerminalIcon className="w-5 h-5" />
                <h2 className="font-semibold text-lg">Execution Log</h2>
            </div>
            {taskStatus && (
                <div className="flex items-center gap-2 text-xs font-mono">
                    <span className={taskStatus.status === 'completed' ? 'text-green-400' : taskStatus.status === 'failed' ? 'text-red-400' : 'text-blue-400'}>
                        {taskStatus.status.toUpperCase()}
                    </span>
                    <span className="text-zinc-600">|</span>
                    <span>{taskId?.split('-')[0]}...</span>
                </div>
            )}
          </div>

          <div className="flex-1 bg-black border border-zinc-800 rounded p-4 font-mono text-xs overflow-y-auto max-h-[500px] shadow-inner">
             {logs.length === 0 && !loading && (
                 <div className="text-zinc-600 text-center mt-20">Ready to start evaluation...</div>
             )}
             {logs.map((log, i) => (
                 <div key={i} className="mb-1 break-all">
                     <span className="text-zinc-500 mr-2">{log.split(']')[0]}]</span>
                     <span className={log.includes('Error') ? 'text-red-400' : log.includes('Warning') ? 'text-yellow-400' : 'text-zinc-300'}>
                        {log.split(']').slice(1).join(']')}
                     </span>
                 </div>
             ))}
             <div ref={logEndRef} />
          </div>

          {/* Progress Bar */}
          {taskStatus && (
              <div className="mt-4 space-y-2">
                  <div className="flex justify-between text-xs text-zinc-400">
                      <span>Progress</span>
                      <span>{Math.round(taskStatus.progress)}%</span>
                  </div>
                  <div className="w-full bg-zinc-800 rounded-full h-1.5">
                      <div 
                        className="bg-white h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${taskStatus.progress}%` }}
                      ></div>
                  </div>
              </div>
          )}
        </div>

        {/* Results Panel (Conditional) */}
        {taskStatus?.status === 'completed' && (
            <div className="glass-panel p-6 rounded-lg border-green-900/30">
                <div className="flex items-center gap-2 mb-4 text-green-400">
                    <CheckCircle2 className="w-5 h-5" />
                    <h2 className="font-semibold text-lg">Evaluation Results</h2>
                </div>
                
                <div className="bg-black/50 p-4 rounded border border-zinc-800">
                    <p className="text-zinc-400 text-sm mb-4">
                        Evaluation finished successfully. The results file is generated.
                    </p>
                    <div className="flex gap-4">
                        <a 
                            href={`${API_BASE}/results/${taskId}`} 
                            target="_blank"
                            className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white text-sm rounded transition-colors"
                        >
                            <FileJson size={16} />
                            View JSON Results
                        </a>
                    </div>
                </div>
            </div>
        )}

        {taskStatus?.status === 'failed' && (
            <div className="glass-panel p-6 rounded-lg border-red-900/30">
                 <div className="flex items-center gap-2 mb-4 text-red-400">
                    <AlertCircle className="w-5 h-5" />
                    <h2 className="font-semibold text-lg">Evaluation Failed</h2>
                </div>
                <p className="text-zinc-400 text-sm">{taskStatus.message}</p>
            </div>
        )}
      </div>
    </div>
  );
}

