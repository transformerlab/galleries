const express = require("express");
const fs = require("fs").promises;
const path = require("path");
const cors = require("cors");

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Set up Pug as the view engine
app.set("view engine", "pug");
app.set("views", path.join(__dirname, "views"));

// Serve static files (for any CSS, JS, images)
app.use(express.static(path.join(__dirname, "public")));

// Path to recipes directory
const RECIPES_DIR = path.join(__dirname, "../../exp-recipes");

// Root route - serve the index page
app.get("/", async (req, res) => {
  try {
    const files = await fs.readdir(RECIPES_DIR);
    const jsonFiles = files.filter((file) => file.endsWith(".json"));

    const recipes = await Promise.all(
      jsonFiles.map(async (file) => {
        try {
          const filePath = path.join(RECIPES_DIR, file);
          const content = await readJsonFile(filePath);
          return {
            filename: file,
            id: content.id || file.replace(".json", ""),
            title: content.title || file.replace(".json", ""),
            description: content.description || "",
            ...content,
          };
        } catch (error) {
          console.error(`Error reading ${file}:`, error.message);
          return {
            filename: file,
            id: file.replace(".json", ""),
            title: file.replace(".json", ""),
            description: "",
            error: error.message,
          };
        }
      })
    );

    res.render("index", { recipes });
  } catch (error) {
    console.error("Error listing recipes:", error);
    res.render("index", { recipes: [], error: "Failed to load recipes" });
  }
});

// Helper function to read and parse JSON file
async function readJsonFile(filePath) {
  try {
    const data = await fs.readFile(filePath, "utf8");
    return JSON.parse(data);
  } catch (error) {
    throw new Error(`Error reading file ${filePath}: ${error.message}`);
  }
}

// Helper function to write JSON file
async function writeJsonFile(filePath, data) {
  try {
    await fs.writeFile(filePath, JSON.stringify(data, null, 2), "utf8");
  } catch (error) {
    throw new Error(`Error writing file ${filePath}: ${error.message}`);
  }
}

// Helper function to check if file exists
async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

// GET /api/recipes - List all recipes with their content
app.get("/api/recipes", async (req, res) => {
  try {
    const files = await fs.readdir(RECIPES_DIR);
    const jsonFiles = files.filter((file) => file.endsWith(".json"));

    const recipes = await Promise.all(
      jsonFiles.map(async (file) => {
        try {
          const filePath = path.join(RECIPES_DIR, file);
          const content = await readJsonFile(filePath);
          return {
            filename: file,
            id: content.id || file.replace(".json", ""),
            title: content.title || file.replace(".json", ""),
            description: content.description || "",
            ...content,
          };
        } catch (error) {
          console.error(`Error reading ${file}:`, error.message);
          return {
            filename: file,
            id: file.replace(".json", ""),
            title: file.replace(".json", ""),
            description: "",
            error: error.message,
          };
        }
      })
    );

    res.json(recipes);
  } catch (error) {
    console.error("Error listing recipes:", error);
    res.status(500).json({ error: "Failed to list recipes" });
  }
});

// GET /api/recipes/:filename - Get a specific recipe
app.get("/api/recipes/:filename", async (req, res) => {
  try {
    const { filename } = req.params;
    const filePath = path.join(RECIPES_DIR, filename);

    if (!(await fileExists(filePath))) {
      return res.status(404).json({ error: "Recipe not found" });
    }

    const content = await readJsonFile(filePath);
    res.json(content);
  } catch (error) {
    console.error("Error reading recipe:", error);
    res.status(500).json({ error: "Failed to read recipe" });
  }
});

// POST /api/recipes - Create a new recipe
app.post("/api/recipes", async (req, res) => {
  try {
    const { filename, content } = req.body;

    if (!filename || !content) {
      return res
        .status(400)
        .json({ error: "Filename and content are required" });
    }

    const filePath = path.join(RECIPES_DIR, filename);

    if (await fileExists(filePath)) {
      return res.status(409).json({ error: "Recipe already exists" });
    }

    await writeJsonFile(filePath, content);
    res.status(201).json({ message: "Recipe created successfully", filename });
  } catch (error) {
    console.error("Error creating recipe:", error);
    res.status(500).json({ error: "Failed to create recipe" });
  }
});

// PUT /api/recipes/:filename - Update an existing recipe
app.put("/api/recipes/:filename", async (req, res) => {
  try {
    const { filename } = req.params;
    const { content } = req.body;

    if (!content) {
      return res.status(400).json({ error: "Content is required" });
    }

    const filePath = path.join(RECIPES_DIR, filename);

    if (!(await fileExists(filePath))) {
      return res.status(404).json({ error: "Recipe not found" });
    }

    await writeJsonFile(filePath, content);
    res.json({ message: "Recipe updated successfully", filename });
  } catch (error) {
    console.error("Error updating recipe:", error);
    res.status(500).json({ error: "Failed to update recipe" });
  }
});

// DELETE /api/recipes/:filename - Delete a recipe
app.delete("/api/recipes/:filename", async (req, res) => {
  try {
    const { filename } = req.params;
    const filePath = path.join(RECIPES_DIR, filename);

    if (!(await fileExists(filePath))) {
      return res.status(404).json({ error: "Recipe not found" });
    }

    await fs.unlink(filePath);
    res.json({ message: "Recipe deleted successfully", filename });
  } catch (error) {
    console.error("Error deleting recipe:", error);
    res.status(500).json({ error: "Failed to delete recipe" });
  }
});

// POST /api/recipes/:filename/duplicate - Duplicate a recipe
app.post("/api/recipes/:filename/duplicate", async (req, res) => {
  try {
    const { filename } = req.params;
    const { newFilename } = req.body;

    if (!newFilename) {
      return res.status(400).json({ error: "New filename is required" });
    }

    const sourcePath = path.join(RECIPES_DIR, filename);
    const targetPath = path.join(RECIPES_DIR, newFilename);

    if (!(await fileExists(sourcePath))) {
      return res.status(404).json({ error: "Source recipe not found" });
    }

    if (await fileExists(targetPath)) {
      return res.status(409).json({ error: "Target recipe already exists" });
    }

    // Read the source file
    const content = await readJsonFile(sourcePath);

    // Update the id and title if they exist
    if (content.id) {
      content.id = newFilename.replace(".json", "");
    }
    if (content.title) {
      content.title = content.title + " (Copy)";
    }

    // Write to the new file
    await writeJsonFile(targetPath, content);

    res.status(201).json({
      message: "Recipe duplicated successfully",
      originalFilename: filename,
      newFilename: newFilename,
    });
  } catch (error) {
    console.error("Error duplicating recipe:", error);
    res.status(500).json({ error: "Failed to duplicate recipe" });
  }
});

// Web routes for creating and editing recipes
app.get("/create", (req, res) => {
  res.render("create");
});

app.get("/edit/:filename", async (req, res) => {
  try {
    const { filename } = req.params;
    const filePath = path.join(RECIPES_DIR, filename);

    if (!(await fileExists(filePath))) {
      return res.status(404).send("Recipe not found");
    }

    const recipe = await readJsonFile(filePath);

    res.render("edit", {
      filename: filename,
      recipe: recipe,
    });
  } catch (error) {
    console.error("Error loading recipe for editing:", error);
    res.status(500).send("Error loading recipe");
  }
});

// Health check endpoint
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// Start server
app.listen(PORT, () => {
  console.log(`Recipe editor server running on port ${PORT}`);
  console.log(`Recipes directory: ${RECIPES_DIR}`);
});

module.exports = app;
