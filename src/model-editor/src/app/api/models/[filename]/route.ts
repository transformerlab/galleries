import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import yaml from 'yaml';
import { ModelConfig } from '@/types/ModelConfig';

const modelsDir = path.join(process.cwd(), '/../../models');

export async function POST(
  request: NextRequest,
  context: { params: { filename: string } }
) {
  const { filename } = await context.params;

  try {
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
      console.error('Parse error:', e);
      return NextResponse.json(
        { error: 'Invalid file content' },
        { status: 400 }
      );
    }

    // Write the file
    await fs.writeFile(filePath, content, 'utf8');
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Failed to save model:', error);
    return NextResponse.json(
      { error: 'Failed to save model' },
      { status: 500 }
    );
  }
}

export async function GET(
  request: NextRequest,
  context: { params: { filename: string } }
) {
  const { filename } = await context.params;

  try {
    const filePath = path.join(modelsDir, filename);
    const fileContent = await fs.readFile(filePath, 'utf-8');
    
    // Parse content based on file type
    let modelData;
    if (filename.endsWith('.json')) {
      modelData = JSON.parse(fileContent);
    } else if (filename.endsWith('.yaml') || filename.endsWith('.yml')) {
      modelData = yaml.parse(fileContent);
    } else {
      return NextResponse.json(
        { error: 'Invalid file type. Must be .json, .yaml, or .yml' },
        { status: 400 }
      );
    }

    return NextResponse.json(modelData);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to load model' },
      { status: 500 }
    );
  }
}

export async function PUT(
  request: NextRequest,
  context: { params: { filename: string } }
) {
  const { filename } = await context.params;

  try {
    const modelData: ModelConfig = await request.json();
    const filePath = path.join(modelsDir, filename);

    console.log('Model data:', modelData);
    
    // Only validate required name field
    // if (!modelData.name) {
    //   return NextResponse.json(
    //     { error: 'Name is required' },
    //     { status: 400 }
    //   );
    // }

    await fs.writeFile(filePath, JSON.stringify(modelData, null, 2));
    return NextResponse.json(modelData);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to update model' },
      { status: 500 }
    );
  }
} 