import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import yaml from 'yaml';

export async function GET() {
  try {
    const modelsDir = path.join(process.cwd(), '/../../models');
    const files = await fs.readdir(modelsDir);
    const modelFiles = files.filter(file => file.endsWith('.json') || file.endsWith('.yaml') || file.endsWith('.yml'));
    
    const models = await Promise.all(modelFiles.map(async (file) => {
      const content = await fs.readFile(path.join(modelsDir, file), 'utf8');
      return {
        filename: file,
        content: file.endsWith('.json') ? JSON.parse(content) : yaml.parse(content)
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
