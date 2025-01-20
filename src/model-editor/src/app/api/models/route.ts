import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

export async function GET() {
  try {
    const modelsDir = path.join(process.cwd(), '/../../models');
    const files = await fs.readdir(modelsDir);
    const jsonFiles = files.filter(file => file.endsWith('.json'));
    
    const models = await Promise.all(jsonFiles.map(async (file) => {
      const content = await fs.readFile(path.join(modelsDir, file), 'utf8');
      return {
        filename: file,
        content: JSON.parse(content)
      };
    }));
    
    return NextResponse.json(models);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch models' },
      { status: 500 }
    );
  }
}
