import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import yaml from 'yaml';
export async function POST(
  request: NextRequest,
  context: { params: { filename: string } }
) {
  const { filename } = await context.params;

  try {
    const modelsDir = path.join(process.cwd(), '/../../models');
    const filePath = path.join(modelsDir, filename);
    
    // Validate that the file exists and is either a .json or .yaml/.yml file
    if (!filename.endsWith('.json') && !filename.endsWith('.yaml') && !filename.endsWith('.yml')) {
      return NextResponse.json(
        { error: 'Invalid file type. Must be .json, .yaml, or .yml' },
        { status: 400 }
      );
    }

    // Get the content from the request body
    const content = await request.text();
    
    // Validate content based on file type
    try {
      if (filename.endsWith('.json')) {
        JSON.parse(content);
      } else if (filename.endsWith('.yaml') || filename.endsWith('.yml')) {
        yaml.parse(content);
      }
    } catch (e) {
      return NextResponse.json(
        { error: 'Invalid file content' },
        { status: 400 }
      );
    }

    // Write the file
    await fs.writeFile(filePath, content, 'utf8');
    
    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to save model' },
      { status: 500 }
    );
  }
} 