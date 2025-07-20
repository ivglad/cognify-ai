# Project: Cognify AI

## General Instructions

- Thoroughly check the attached sources and instructions - they have the highest priority.
- Strive to use the most up-to-date information (as of 2025) using web search, MCP, if this information does not contradict the attached sources.
- Analyze the project and all necessary files before each response.
- If the user asks to write a plan, wait for confirmation of actions after its description.
- Automatically fix linting errors.

## Sources

- **Infinity**
- **Kuzu**

## Tech Stack

- **Backend:** Python 3.11+, FastAPI 0.100+
- **Vector Store:** [Infinity | infiniflow.org]()
- **RAG:** Used in document processing
- **Orchestration LLM:** LangChain (Chain, Agent, Memory)
- **Embedding Models:** Yandex-embedder, Giga-embedder, and OpenRouter for LLM access
- **Hybrid Search:** Hybrid search using nltk + spacy + Infinity + Kuzu
- **Knowledge Graph:** [Kuzu | kuzudb.com]()
- **Frontend:** Tauri 2.6 + Vue 3.5 + Pinia + Vue Router + WebSocket API (future)

## Code Commenting Rules

All code comments must comply with the following rules:

- All comments must be in **RUSSIAN**.
- Comments must be placed above each function, block of non-obvious logic, or block requiring description.
- Comments must be concise and convey the essence of the logic.
- Comments must be single-line.
- Comments should not describe "why" it was created, only the logic.

### Examples of Code Commenting:

**Incorrect:**

- "Creating a full-text index"
- "Changing query logic"
- "Using an asynchronous method for embeddings"
- "Ensuring default database existence"

**Correct:**

- "Full-text search"
- "Query logic"
- "Asynchronous method for embeddings"
- "Default database"

## Git Rules

- Never execute git commands (git add, commit, pull, push, etc.). Only the user can do this.

# Clean Code Guidelines

## Constants Over Magic Numbers

- Replace hard-coded values with named constants
- Use descriptive constant names that explain the value's purpose
- Keep constants at the top of the file or in a dedicated constants file

## Meaningful Names

- Variables, functions, and classes should reveal their purpose
- Names should explain why something exists and how it's used
- Avoid abbreviations unless they're universally understood

## Smart Comments

- Don't comment on what the code does - make the code self-documenting
- Use comments to explain why something is done a certain way
- Document APIs, complex algorithms, and non-obvious side effects

## Single Responsibility

- Each function should do exactly one thing
- Functions should be small and focused
- If a function needs a comment to explain what it does, it should be split

## DRY (Don't Repeat Yourself)

- Extract repeated code into reusable functions
- Share common logic through proper abstraction
- Maintain single sources of truth

## Clean Structure

- Keep related code together
- Organize code in a logical hierarchy
- Use consistent file and folder naming conventions

## Encapsulation

- Hide implementation details
- Expose clear interfaces
- Move nested conditionals into well-named functions

## Code Quality Maintenance

- Refactor continuously
- Fix technical debt early
- Leave code cleaner than you found it

## Testing

- Write tests before fixing bugs
- Keep tests readable and maintainable
- Test edge cases and error conditions

## Version Control

- Write clear commit messages
- Make small, focused commits
- Use meaningful branch names

# Code Editing Principles

## Golden Rule

Treat every edit as a “surgical, precise replacement,” not a “fuzzy adjustment.”

## Principle 1: Replace, Don’t Modify

**Action Guideline**: Target the smallest, complete logical block and provide a brand new block to completely replace it.

**DO**:

- Replace entire lines or whole methods/logical blocks.
  ```csharp
  // ... existing code ...
  public int CalculateTotalPrice() { /* new, improved logic */ }
  // ... existing code ...
  ```

**DON’T**:

- Try to only change part of a line or just rename a method in place.
  ```csharp
  // ... existing code ...
  CalculateNewTotalPrice() // Only changing the method name—very risky
  // ... existing code ...
  ```

## Principle 2: Anchors Must Be Unique

**Action Guideline**: The context you use as anchors (e.g., `// ... existing code ...`) must be unique within the file, just like a fingerprint.

**DO**:

- Include enough surrounding context to ensure uniqueness.
  ```csharp
  // ... existing code ...
  var result = await _service.GetSpecificData(id);
  return View(result); // <--- This context combination is likely unique
  // ... existing code ...
  ```

**DON’T**:

- Use generic or potentially repeated code as an anchor (e.g., just a closing bracket).
  ```csharp
  // ... existing code ...
  } // <--- Highly likely to be a repeated anchor
  // ... existing code ...
  ```

## Principle 3: Code Must Be Complete

**Action Guideline**: The submitted `code_edit` must be a syntactically correct, logically self-contained unit. Don’t make the model guess.

**DO**:

- Make sure the code you submit can be copy-pasted into the IDE without syntax errors.

**DON’T**:

- Submit incomplete statements.
  ```csharp
  // ... existing code ...
  var user = new User { Name = // <--- Incomplete code
  // ... existing code ...
  ```

## Principle 4: Decompose Complex Tasks

**Action Guideline**: Large refactors = multiple consecutive, simple, and safe small replacements.

**DO**:

- First `edit_file`: Add a new helper method.
- Second `edit_file`: Replace the old logic block with code that calls the new method.

**DON’T**:

- Define a new method and change all its usages in multiple places within a single `edit_file` operation.

## Principle 5: Instructions Must Accurately Describe

**Action Guideline**: The `instructions` parameter should be a one-sentence, precise summary of the `code_edit`.

**DO**:

- Example: `instructions`: "I will replace the user validation logic with a call to the new AuthService."

**DON’T**:

- Use vague instructions like `instructions`: "Fix bug" or `instructions`: "Update code" (too broad and not helpful).
