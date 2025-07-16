# Recipe Editor Server

A Node.js Express server for managing recipe files in the galleries workspace.

## Features

- List all recipes with their content
- Load individual recipe files
- Create new recipes
- Update existing recipes
- Delete recipes
- Duplicate recipes

## Installation

```bash
npm install
```

## Running the Server

```bash
# Development mode with auto-reload
npm run dev

# Production mode
npm start
```

The server will run on port 3001 by default (configurable via PORT environment variable).

## API Endpoints

### List All Recipes
```
GET /api/recipes
```
Returns an array of all recipes with their full content, including title, description, and other metadata.

### Get Specific Recipe
```
GET /api/recipes/:filename
```
Returns the content of a specific recipe file.

### Create New Recipe
```
POST /api/recipes
```
Body:
```json
{
  "filename": "my-recipe.json",
  "content": {
    "id": "my-recipe",
    "title": "My Recipe",
    "description": "Recipe description",
    ...
  }
}
```

### Update Recipe
```
PUT /api/recipes/:filename
```
Body:
```json
{
  "content": {
    "id": "my-recipe",
    "title": "Updated Recipe",
    "description": "Updated description",
    ...
  }
}
```

### Delete Recipe
```
DELETE /api/recipes/:filename
```

### Duplicate Recipe
```
POST /api/recipes/:filename/duplicate
```
Body:
```json
{
  "newFilename": "my-recipe-copy.json"
}
```

### Health Check
```
GET /api/health
```
Returns server status and timestamp.

## Error Handling

The server returns appropriate HTTP status codes:
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `404` - Not Found
- `409` - Conflict (file already exists)
- `500` - Internal Server Error

## Directory Structure

The server manages JSON files in the `../../exp-recipes/` directory relative to the server location.

## Example Usage

```javascript
// List all recipes
const recipes = await fetch('/api/recipes').then(r => r.json());

// Get a specific recipe
const recipe = await fetch('/api/recipes/my-recipe.json').then(r => r.json());

// Create a new recipe
await fetch('/api/recipes', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    filename: 'new-recipe.json',
    content: { id: 'new-recipe', title: 'New Recipe' }
  })
});

// Update a recipe
await fetch('/api/recipes/my-recipe.json', {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    content: { id: 'my-recipe', title: 'Updated Title' }
  })
});

// Delete a recipe
await fetch('/api/recipes/my-recipe.json', { method: 'DELETE' });

// Duplicate a recipe
await fetch('/api/recipes/my-recipe.json/duplicate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ newFilename: 'my-recipe-copy.json' })
});
```
