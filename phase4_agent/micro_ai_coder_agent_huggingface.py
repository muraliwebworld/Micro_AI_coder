#!/usr/bin/env python3
"""
PHASE 3: REACT COMPONENT INFERENCE (HUGGING FACE MODEL VERSION)
Generates React components using the HF-trained model.
"""

import torch
import torch.nn.functional as F
import json
import tiktoken
import re
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
LOGS_DIR = Path(__file__).parent.parent / "logs"
MODEL_PATH = MODELS_DIR / "tiny_code_model_huggingface.pt"
CONFIG_PATH = MODELS_DIR / "model_config_huggingface.json"
ERROR_LOG_FILE = LOGS_DIR / "inference_errors_huggingface.jsonl"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_header(text):
    print("\n" + "="*80)
    print(text)
    print("="*80)

def print_step(text):
    print(f"\n➤ {text}")

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

def print_warning(text):
    print(f"⚠️  {text}")

def log_generation_error(prompt, generated_code, validation_errors):
    """Log generated code that failed validation"""
    error_entry = {
        'timestamp': datetime.now().isoformat(),
        'prompt': prompt,
        'generated_code': generated_code,
        'validation_errors': validation_errors,
        'code_length': len(generated_code),
        'model': 'huggingface'
    }
    
    try:
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print_warning(f"Could not write error log: {str(e)}")
        return False

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_config():
    """Load trained HF model and configuration"""
    print_step("Loading Hugging Face model and configuration...")
    
    if not CONFIG_PATH.exists() or not MODEL_PATH.exists():
        print_error("Model or config not found!")
        print("Run training first: python phase2_training/v2_train_model_huggingface.py")
        sys.exit(1)
    
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print_success(f"Config loaded (HF dataset)")
    print(f"  - Embedding: {config['n_embd']}")
    print(f"  - Heads: {config['n_head']}")
    print(f"  - Layers: {config['n_layer']}")
    print(f"  - Vocab: {config['vocab_size']:,}")
    print(f"  - Best val loss: {config.get('best_val_loss', 'N/A')}")
    
    enc = tiktoken.get_encoding(config['tokenizer'])
    print_success(f"Tokenizer loaded ({config['tokenizer']})")
    
    return config, enc

# ============================================================================
# CODE GENERATOR CLASS
# ============================================================================

class CodeGenerator:
    """Generate React components with HF-trained model"""
    
    def __init__(self, config, enc, device=DEVICE):
        self.config = config
        self.enc = enc
        self.device = device
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained model from disk"""
        from phase2_training.v2_train_model_huggingface import TinyCodeModel
        
        model = TinyCodeModel(
            vocab_size=self.config['vocab_size'],
            n_embd=self.config['n_embd'],
            n_head=self.config['n_head'],
            n_layer=self.config['n_layer'],
            block_size=self.config['block_size'],
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=False))
        model.eval()
        self.model = model
        print_success("HF Model loaded")
    
    def validate_react_component(self, code):
        """Validate if generated code is valid React"""
        if not code or len(code) < 100:
            return False, ["Code too short"]
        
        errors = []
        
        # Balance checks
        open_parens = code.count('(') - code.count(')')
        open_braces = code.count('{') - code.count('}')
        open_brackets = code.count('[') - code.count(']')
        
        if open_parens != 0:
            errors.append(f"Unbalanced parentheses (diff: {open_parens})")
        if open_braces != 0:
            errors.append(f"Unbalanced braces (diff: {open_braces})")
        if open_brackets != 0:
            errors.append(f"Unbalanced brackets (diff: {open_brackets})")
        
        if 'import React' not in code:
            errors.append("Missing 'import React'")
        
        if 'export default' not in code and 'module.exports' not in code:
            errors.append("Missing export statement")
        
        corruption_patterns = [
            (r'import React\)', 'Corrupted import'),
            (r'const \[.+?\],', 'Malformed state'),
            (r'useState\(.*?,.*?,', 'Multiple commas in useState'),
            (r'PropTypes\),', 'Corrupted PropTypes'),
        ]
        
        for pattern, error_msg in corruption_patterns:
            if re.search(pattern, code):
                errors.append(error_msg)
        
        if not re.search(r'(const|function|class)\s+\w+\s*=?\s*(\(|{)', code):
            errors.append("No valid component definition")
        
        return len(errors) == 0, errors

    def generate_react_component(self, prompt, temperature=0.6, top_k=40, max_tokens=800):
        """Generate React component from prompt"""
        print_step(f"Generating React component...")
        
        component_prompt = f"React component for: {prompt}\n\nimport React"
        tokens = self.enc.encode(component_prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        generated_code = component_prompt
        
        with torch.no_grad():
            for token_idx in range(max_tokens):
                logits = self.model(x[:, -self.config['block_size']:])
                logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                token_str = self.enc.decode([idx_next.item()])
                generated_code += token_str
                
                x = torch.cat((x, idx_next), dim=1)
                
                if '<|file_end|>' in token_str or token_idx > max_tokens:
                    break
        
        generated_code = generated_code.replace('<|file_end|>', '').strip()
        
        is_valid, error_messages = self.validate_react_component(generated_code)
        
        if is_valid:
            print_success(f"Generated component ({len(generated_code)} chars)")
            return generated_code
        else:
            error_str = "; ".join(error_messages)
            print_warning(f"Generated code failed validation: {error_str}")
            print_warning("Using fallback template...")
            
            log_generation_error(prompt, generated_code, error_messages)
            return self._get_fallback_component(prompt)
    
    def _get_fallback_component(self, prompt):
        """Fallback template based on prompt"""
        prompt_lower = prompt.lower()
        
        if 'form' in prompt_lower or 'login' in prompt_lower or 'signup' in prompt_lower:
            return """import React, { useState } from 'react';

const LoginForm = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please fill in all fields');
      return;
    }
    if (!email.includes('@')) {
      setError('Please enter a valid email');
      return;
    }
    console.log('Form submitted:', { email, password });
    alert('Login successful!');
  };
  
  return (
    <div style={{ maxWidth: '400px', margin: '50px auto', padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h2>Login</h2>
      {error && <div style={{ color: 'red', marginBottom: '10px' }}>{error}</div>}
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '15px' }}>
          <label>Email:</label>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Enter email" style={{ width: '100%', padding: '8px', marginTop: '5px' }} />
        </div>
        <div style={{ marginBottom: '15px' }}>
          <label>Password:</label>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Enter password" style={{ width: '100%', padding: '8px', marginTop: '5px' }} />
        </div>
        <button type="submit" style={{ width: '100%', padding: '10px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>Login</button>
      </form>
    </div>
  );
};

export default LoginForm;"""
        
        elif 'counter' in prompt_lower:
            return """import React, { useState } from 'react';

const Counter = () => {
  const [count, setCount] = useState(0);
  
  return (
    <div style={{ textAlign: 'center', padding: '20px' }}>
      <h2>Counter</h2>
      <div style={{ fontSize: '48px', fontWeight: 'bold', margin: '20px' }}>{count}</div>
      <button onClick={() => setCount(count + 1)} style={{ padding: '10px 20px', margin: '5px' }}>Increment</button>
      <button onClick={() => setCount(count - 1)} style={{ padding: '10px 20px', margin: '5px' }}>Decrement</button>
      <button onClick={() => setCount(0)} style={{ padding: '10px 20px', margin: '5px', backgroundColor: '#ff6b6b', color: 'white' }}>Reset</button>
    </div>
  );
};

export default Counter;"""
        
        elif 'todo' in prompt_lower or 'list' in prompt_lower:
            return """import React, { useState } from 'react';

const TodoList = () => {
  const [todos, setTodos] = useState([]);
  const [input, setInput] = useState('');
  
  const addTodo = () => {
    if (input.trim()) {
      setTodos([...todos, { id: Date.now(), text: input, completed: false }]);
      setInput('');
    }
  };
  
  return (
    <div style={{ maxWidth: '500px', margin: '50px auto', padding: '20px' }}>
      <h2>Todo List</h2>
      <div style={{ marginBottom: '15px' }}>
        <input type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Add todo..." style={{ width: '80%', padding: '8px' }} />
        <button onClick={addTodo} style={{ padding: '8px 15px', marginLeft: '5px' }}>Add</button>
      </div>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {todos.map(todo => (
          <li key={todo.id} style={{ padding: '10px', borderBottom: '1px solid #ddd' }}>
            {todo.text}
            <button onClick={() => setTodos(todos.filter(t => t.id !== todo.id))} style={{ marginLeft: '10px', color: 'red', border: 'none', cursor: 'pointer', backgroundColor: '#fff' }}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default TodoList;"""
        
        else:
            return """import React, { useState } from 'react';

const Component = () => {
  const [state, setState] = useState(null);
  
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>React Component</h2>
      <input type="text" value={state || ''} onChange={(e) => setState(e.target.value)} placeholder="Enter something..." style={{ padding: '8px', width: '100%', maxWidth: '400px', marginBottom: '10px' }} />
      {state && <div style={{ padding: '10px', backgroundColor: '#f0f0f0', borderRadius: '4px', marginTop: '10px' }}><p><strong>You entered:</strong> {state}</p></div>}
    </div>
  );
};

export default Component;"""
    
    def generate_component_explanation(self, prompt, code):
        """Generate explanation for component"""
        print_step("Generating explanation...")
        
        cot = f"This component implements: {prompt[:80]}. "
        
        if 'form' in prompt.lower():
            cot += "It includes form handling with input fields and validation. "
        if 'list' in prompt.lower() or 'todo' in prompt.lower():
            cot += "It manages a list of items with add/remove functionality. "
        if 'counter' in prompt.lower():
            cot += "It provides increment and decrement operations. "
        
        if 'useState' in code:
            cot += "State management is handled with React hooks. "
        
        cot += "The component is a functional React component with proper JSX structure."
        
        print_success(f"Generated explanation ({len(cot)} chars)")
        return cot

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def generate_single_component(generator, prompt):
    """Generate single component"""
    code = generator.generate_react_component(prompt)
    cot = generator.generate_component_explanation(prompt, code)
    
    return {
        "prompt": prompt,
        "code": code,
        "Complex_CoT": cot
    }

def save_component(component_dict, output_dir):
    """Save component to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'prompt': component_dict['prompt'],
        'model': 'huggingface'
    }
    
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output_dir / "component.jsx").write_text(component_dict['code'], encoding='utf-8')
    (output_dir / "explanation.md").write_text(f"# Component Explanation\n\n{component_dict['Complex_CoT']}", encoding='utf-8')
    
    print_success(f"Saved to {output_dir}/")
    print(f"  - component.jsx ({len(component_dict['code'])} chars)")
    print(f"  - explanation.md ({len(component_dict['Complex_CoT'])} chars)")

def interactive_menu(generator):
    """Interactive menu"""
    while True:
        print_header("REACT COMPONENT INFERENCE - HUGGING FACE MODEL")
        print("\n1. Generate a React component")
        print("2. Generate multiple components")
        print("3. View error logs")
        print("4. Exit")
        print("\n➤ Choice (1-4): ", end="")
        
        choice = input().strip()
        
        if choice == '1':
            print("\n💬 Describe a React component:")
            print("➤ Prompt: ", end="")
            prompt = input().strip()
            
            if not prompt:
                print_error("Empty prompt")
                continue
            
            print(f"\n🔄 Generating component...\n")
            
            try:
                component = generate_single_component(generator, prompt)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = OUTPUTS_DIR / f"{timestamp}_hf"
                save_component(component, output_dir)
                
                print_header("✅ COMPONENT GENERATED")
                print(f"Location: {output_dir}")
                
            except Exception as e:
                print_error(f"Generation failed: {str(e)}")
        
        elif choice == '2':
            print("\n📋 Generate multiple components")
            print("➤ How many? (1-10): ", end="")
            try:
                count = int(input().strip())
                count = max(1, min(10, count))
            except:
                count = 3
            
            prompts = []
            print(f"\nEnter {count} prompts:")
            for i in range(count):
                print(f"  {i+1}. ", end="")
                prompts.append(input().strip())
            
            for idx, prompt in enumerate(prompts, 1):
                if not prompt:
                    print_warning(f"Skipping empty prompt {idx}")
                    continue
                
                try:
                    component = generate_single_component(generator, prompt)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = OUTPUTS_DIR / f"{timestamp}_hf_comp{idx}"
                    save_component(component, output_dir)
                    print(f"  [{idx}/{count}] ✅ Generated")
                except Exception as e:
                    print_error(f"  [{idx}/{count}] Failed: {str(e)}")
        
        elif choice == '3':
            if not ERROR_LOG_FILE.exists():
                print_warning("No error logs")
            else:
                with open(ERROR_LOG_FILE) as f:
                    lines = f.readlines()
                print_header("ERROR LOGS")
                print(f"Total errors: {len(lines)}")
                for error_line in lines[-5:]:
                    data = json.loads(error_line)
                    print(f"\n  Prompt: {data['prompt'][:50]}")
                    print(f"  Errors: {'; '.join(data['validation_errors'])}")
        
        elif choice == '4':
            print_success("Goodbye!")
            break
        
        else:
            print_error("Invalid choice")

def main():
    config, enc = load_model_and_config()
    generator = CodeGenerator(config, enc, device=DEVICE)
    interactive_menu(generator)

if __name__ == "__main__":
    main()
