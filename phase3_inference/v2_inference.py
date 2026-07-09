#!/usr/bin/env python3
"""
PHASE 3: REACT COMPONENT INFERENCE
Generates single React components from prompts using the trained model.
Input: Trained model + tiktoken tokenizer + component prompt
Output: React component code + Chain-of-Thought explanation
"""

import torch
import torch.nn.functional as F
import json
import tiktoken
import re
from pathlib import Path
from datetime import datetime
import sys

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
LOGS_DIR = Path(__file__).parent.parent / "logs"
MODEL_PATH = MODELS_DIR / "tiny_code_model.pt"
CONFIG_PATH = MODELS_DIR / "model_config.json"
ERROR_LOG_FILE = LOGS_DIR / "inference_errors.jsonl"

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
    """Log generated code that failed validation to file"""
    error_entry = {
        'timestamp': datetime.now().isoformat(),
        'prompt': prompt,
        'generated_code': generated_code,
        'validation_errors': validation_errors,
        'code_length': len(generated_code)
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
    """Load trained model and configuration"""
    print_step("Loading model and configuration...")
    
    if not CONFIG_PATH.exists() or not MODEL_PATH.exists():
        print_error("Model or config not found!")
        print("Run Phase 2 first: python phase2_training/v2_train_model.py")
        sys.exit(1)
    
    # Load config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print_success(f"Config loaded")
    print(f"  - Embedding: {config['n_embd']}")
    print(f"  - Heads: {config['n_head']}")
    print(f"  - Layers: {config['n_layer']}")
    print(f"  - Vocab: {config['vocab_size']:,}")
    
    # Load tokenizer
    enc = tiktoken.get_encoding(config['tokenizer'])
    print_success(f"Tokenizer loaded ({config['tokenizer']})")
    
    return config, enc

# ============================================================================
# CODE GENERATOR CLASS
# ============================================================================

class CodeGenerator:
    """Generate React components with token-by-token streaming"""
    
    def __init__(self, config, enc, device=DEVICE):
        self.config = config
        self.enc = enc
        self.device = device
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained model from disk"""
        from phase2_training.v2_train_model import TinyCodeModel
        
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
        print_success("Model loaded")
    
    def validate_react_component(self, code):
        """
        Validate if generated code looks like valid React code.
        Returns: (is_valid, error_message_list)
        """
        if not code or len(code) < 100:
            return False, ["Code too short"]
        
        errors = []
        
        # Count parentheses/brackets/braces
        open_parens = code.count('(') - code.count(')')
        open_braces = code.count('{') - code.count('}')
        open_brackets = code.count('[') - code.count(']')
        
        if open_parens != 0:
            errors.append(f"Unbalanced parentheses (diff: {open_parens})")
        if open_braces != 0:
            errors.append(f"Unbalanced braces (diff: {open_braces})")
        if open_brackets != 0:
            errors.append(f"Unbalanced brackets (diff: {open_brackets})")
        
        # Check for valid imports
        if 'import React' not in code:
            errors.append("Missing 'import React'")
        
        # Check for valid export
        if 'export default' not in code and 'module.exports' not in code:
            errors.append("Missing export statement")
        
        # Check for common corruption patterns
        corruption_patterns = [
            (r'import React\)', 'Corrupted import statement'),
            (r'const \[.+?\],', 'Malformed state declaration'),
            (r'useState\(.*?,.*?,', 'Multiple commas in useState'),
            (r'undefined.*undefined', 'Multiple undefined tokens'),
            (r'PropTypes\),', 'Corrupted PropTypes'),
        ]
        
        for pattern, error_msg in corruption_patterns:
            if re.search(pattern, code):
                errors.append(error_msg)
        
        # Must have a function or class definition
        if not re.search(r'(const|function|class)\s+\w+\s*=?\s*(\(|{)', code):
            errors.append("No valid component definition found")
        
        if errors:
            return False, errors
        
        return True, []

    def generate_react_component(self, prompt, temperature=0.4, top_k=30, max_tokens=600):
        """
        Generate a React component from a prompt using the trained model.
        Returns component code or falls back to template if validation fails.
        
        IMPROVED GENERATION:
        - Lower temperature (0.4) for more focused/conservative generation
        - Smaller top_k (30) to reduce noise
        - Shorter max_tokens to prevent divergence
        - Better prompt engineering with React-specific context
        """
        print_step(f"Generating React component...")
        
        # Enhanced prompt for component generation - more specific context
        component_prompt = f"""import React, {{ useState }} from 'react';

/**
 * {prompt}
 */
const Component = () => {{"""
        
        # Encode prompt
        tokens = self.enc.encode(component_prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        generated_code = component_prompt
        bracket_count = 0
        
        with torch.no_grad():
            for token_idx in range(max_tokens):
                # Get next token logits
                logits = self.model(x[:, -self.config['block_size']:])
                logits = logits[:, -1, :] / temperature
                
                # Top-k filtering with more aggressive cutoff
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Add small penalty to tokens that break structure
                bad_token_strs = ['undefined', 'null', '<|file_end|>']
                for bad_str in bad_token_strs:
                    try:
                        bad_token = self.enc.encode(bad_str)[0]
                        logits[0, bad_token] -= 1.0
                    except:
                        pass
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Decode token
                token_str = self.enc.decode([idx_next.item()])
                generated_code += token_str
                
                # Track brackets to detect end of component
                bracket_count += token_str.count('{') - token_str.count('}')
                
                # Append to sequence
                x = torch.cat((x, idx_next), dim=1)
                
                # Stop conditions
                if '<|file_end|>' in token_str:
                    break
                if 'export default' in generated_code and bracket_count == 0:
                    break
        
        # Clean up generated code
        generated_code = generated_code.replace('<|file_end|>', '').strip()
        
        # Ensure proper closing
        if not generated_code.endswith(';'):
            generated_code += ';\n'
        
        # Validate generated code
        is_valid, error_messages = self.validate_react_component(generated_code)
        
        if is_valid:
            print_success(f"✨ Generated valid component ({len(generated_code)} chars)")
            return generated_code
        else:
            # Log the error for analysis
            error_str = "; ".join(error_messages)
            print_warning(f"Generated code failed validation: {error_str}")
            print_warning("Using fallback template instead...")
            
            # Save error log for analysis
            log_generation_error(prompt, generated_code, error_messages)
            
            # Use fallback
            return self._get_fallback_component(prompt)
    
    def _get_fallback_component(self, prompt):
        """Fallback template when model generation fails validation"""
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
    setError('');
    alert('Login successful!');
  };
  
  return (
    <div style={{ maxWidth: '400px', margin: '50px auto', padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h2>Login</h2>
      {error && <div style={{ color: 'red', marginBottom: '10px' }}>{error}</div>}
      
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '15px' }}>
          <label>Email:</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email"
            style={{ width: '100%', padding: '8px', marginTop: '5px', boxSizing: 'border-box' }}
          />
        </div>
        
        <div style={{ marginBottom: '15px' }}>
          <label>Password:</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter your password"
            style={{ width: '100%', padding: '8px', marginTop: '5px', boxSizing: 'border-box' }}
          />
        </div>
        
        <button
          type="submit"
          style={{ width: '100%', padding: '10px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '16px' }}
        >
          Login
        </button>
      </form>
    </div>
  );
};

export default LoginForm;"""
        
        elif 'counter' in prompt_lower or 'increment' in prompt_lower or 'decrement' in prompt_lower:
            return """import React, { useState } from 'react';

const Counter = () => {
  const [count, setCount] = useState(0);
  
  const increment = () => setCount(count + 1);
  const decrement = () => setCount(count - 1);
  const reset = () => setCount(0);
  
  return (
    <div style={{ textAlign: 'center', padding: '20px' }}>
      <h2>Counter</h2>
      <div style={{ fontSize: '48px', fontWeight: 'bold', margin: '20px' }}>
        {count}
      </div>
      
      <div>
        <button
          onClick={increment}
          style={{ padding: '10px 20px', margin: '5px', fontSize: '16px', cursor: 'pointer' }}
        >
          Increment
        </button>
        
        <button
          onClick={decrement}
          style={{ padding: '10px 20px', margin: '5px', fontSize: '16px', cursor: 'pointer' }}
        >
          Decrement
        </button>
        
        <button
          onClick={reset}
          style={{ padding: '10px 20px', margin: '5px', fontSize: '16px', cursor: 'pointer', backgroundColor: '#ff6b6b', color: 'white' }}
        >
          Reset
        </button>
      </div>
    </div>
  );
};

export default Counter;"""
        
        elif 'todo' in prompt_lower or 'list' in prompt_lower or 'task' in prompt_lower:
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
  
  const toggleTodo = (id) => {
    setTodos(todos.map(todo =>
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  };
  
  const removeTodo = (id) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };
  
  return (
    <div style={{ maxWidth: '500px', margin: '50px auto', padding: '20px' }}>
      <h2>Todo List</h2>
      
      <div style={{ marginBottom: '15px' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && addTodo()}
          placeholder="Add a new todo..."
          style={{ width: '80%', padding: '8px', marginRight: '5px' }}
        />
        <button
          onClick={addTodo}
          style={{ padding: '8px 15px', cursor: 'pointer' }}
        >
          Add
        </button>
      </div>
      
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {todos.map(todo => (
          <li key={todo.id} style={{ padding: '10px', borderBottom: '1px solid #ddd', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span
              onClick={() => toggleTodo(todo.id)}
              style={{
                textDecoration: todo.completed ? 'line-through' : 'none',
                cursor: 'pointer',
                flex: 1
              }}
            >
              {todo.text}
            </span>
            <button
              onClick={() => removeTodo(todo.id)}
              style={{ color: 'red', border: 'none', cursor: 'pointer', backgroundColor: '#fff' }}
            >
              Delete
            </button>
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
  
  const handleChange = (e) => {
    setState(e.target.value);
  };
  
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>React Component</h2>
      
      <input 
        type="text"
        value={state || ''}
        onChange={handleChange}
        placeholder="Enter something..."
        style={{ padding: '8px', width: '100%', maxWidth: '400px', marginBottom: '10px' }}
      />
      
      {state && (
        <div style={{ padding: '10px', backgroundColor: '#f0f0f0', borderRadius: '4px', marginTop: '10px' }}>
          <p><strong>You entered:</strong> {state}</p>
        </div>
      )}
    </div>
  );
};

export default Component;"""
    
    def generate_component_explanation(self, prompt, code):
        """
        Generate a Chain-of-Thought explanation for the component.
        This mimics the Complex_CoT field in the training dataset.
        """
        print_step("Generating explanation...")
        
        # Use a heuristic-based explanation
        cot_template = f"This component implements: {prompt[:80]}. "
        
        # Detect key patterns and add explanation
        if 'form' in prompt.lower():
            cot_template += "It includes input handling and state management with useState hooks. "
        if 'list' in prompt.lower():
            cot_template += "It renders items from an array using map() function. "
        if 'button' in prompt.lower():
            cot_template += "It includes onClick handlers for user interactions. "
        if 'api' in prompt.lower() or 'fetch' in prompt.lower():
            cot_template += "It includes useEffect for data fetching from APIs. "
        
        if 'useState' in code:
            cot_template += "State management is handled with React hooks (useState). "
        if 'useEffect' in code:
            cot_template += "Side effects are managed with useEffect hook. "
        
        cot_template += "The component is a functional React component with proper JSX structure."
        
        print_success(f"Generated explanation ({len(cot_template)} chars)")
        return cot_template
    
    def parse_user_prompt(self, user_prompt):
        """
        Simple prompt parsing for React component generation.
        Returns component description and styling preference.
        """
        prompt_lower = user_prompt.lower()
        
        # Extract styling framework
        styling = 'tailwind'
        if 'bootstrap' in prompt_lower:
            styling = 'bootstrap'
        elif 'css' in prompt_lower or 'style' in prompt_lower or 'styled' in prompt_lower:
            styling = 'css'
        elif 'material' in prompt_lower or 'mui' in prompt_lower:
            styling = 'material'
        
        return {
            'component_description': user_prompt,
            'styling': styling
        }

# ============================================================================
# MAIN INFERENCE FUNCTIONS
# ============================================================================

def generate_single_component(generator, prompt):
    """Generate a single React component and return as dict"""
    code = generator.generate_react_component(prompt)
    cot = generator.generate_component_explanation(prompt, code)
    
    return {
        "prompt": prompt,
        "code": code,
        "Complex_CoT": cot
    }

def save_component(component_dict, output_dir):
    """Save component to files in output directory"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'prompt': component_dict['prompt']
    }
    
    metadata_file = output_dir / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    
    # Save component code
    component_file = output_dir / "component.jsx"
    component_file.write_text(component_dict['code'], encoding='utf-8')
    
    # Save explanation
    explanation_file = output_dir / "explanation.md"
    explanation_file.write_text(f"# Component Explanation\n\n{component_dict['Complex_CoT']}", encoding='utf-8')
    
    print_success(f"Saved to {output_dir}/")
    print(f"  - component.jsx ({len(component_dict['code'])} chars)")
    print(f"  - explanation.md ({len(component_dict['Complex_CoT'])} chars)")

# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def interactive_menu(generator):
    """Interactive testing menu"""
    while True:
        print_header("PHASE 3: REACT COMPONENT INFERENCE")
        print("\n1. Generate a React component")
        print("2. Generate multiple components")
        print("3. View error logs")
        print("4. Exit")
        print("\n➤ Choice (1-4): ", end="")
        
        choice = input().strip()
        
        if choice == '1':
            print("\n💬 Describe a React component (what should it do?):")
            print("\nExamples:")
            print("  • Generate a React component for a login form")
            print("  • Create a React counter component with increment and decrement buttons")
            print("  • Write a React todo list component with add/remove functionality")
            print("  • Create a React form with email and password validation")
            print("  • Build a React product card component with image and rating")
            print("\n➤ Component description: ", end="")
            prompt = input().strip()
            
            if not prompt:
                print_error("Empty prompt, skipping...")
                continue
            
            print(f"\n🔄 Starting component generation...\n")
            
            try:
                component = generate_single_component(generator, prompt)
                
                # Save output
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = OUTPUTS_DIR / timestamp
                save_component(component, output_dir)
                
                print_header("✅ COMPONENT GENERATED SUCCESSFULLY")
                print(f"\n📁 Location: {output_dir}")
                
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
            print(f"\nEnter {count} component prompts:")
            for i in range(count):
                print(f"  {i+1}. ", end="")
                prompts.append(input().strip())
            
            print(f"\n🔄 Generating {count} components...\n")
            
            for idx, prompt in enumerate(prompts, 1):
                if not prompt:
                    print_warning(f"Skipping empty prompt {idx}")
                    continue
                
                try:
                    component = generate_single_component(generator, prompt)
                    
                    # Save output
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = OUTPUTS_DIR / f"{timestamp}_comp{idx}"
                    save_component(component, output_dir)
                    
                    print(f"  [{idx}/{count}] ✅ Generated: {prompt[:50]}")
                    
                except Exception as e:
                    print_error(f"  [{idx}/{count}] Failed: {str(e)}")
            
            print_success(f"\n✅ Generated {count} components!")
        
        elif choice == '3':
            # View error logs
            if not ERROR_LOG_FILE.exists():
                print_warning("No error logs found yet")
            else:
                print_header("📋 GENERATION ERROR LOGS")
                with open(ERROR_LOG_FILE, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print(f"\nTotal errors: {len(lines)}")
                    print("\nRecent errors (last 5):")
                    for error_line in lines[-5:]:
                        error_data = json.loads(error_line)
                        print(f"\n  Prompt: {error_data['prompt'][:60]}")
                        print(f"  Code length: {error_data['code_length']} chars")
                        print(f"  Errors: {'; '.join(error_data['validation_errors'])}")
        
        elif choice == '4':
            print_success("Goodbye!")
            break
        
        else:
            print_error("Invalid choice")

def main():
    """Main entry point"""
    config, enc = load_model_and_config()
    generator = CodeGenerator(config, enc, device=DEVICE)
    interactive_menu(generator)

if __name__ == "__main__":
    main()
