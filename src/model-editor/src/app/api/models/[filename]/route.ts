import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

export async function POST(
  request: NextRequest,
  context: { params: { filename: string } }
) {
  const { filename } = await context.params;

  try {
    const modelsDir = path.join(process.cwd(), '/../../models');
    const filePath = path.join(modelsDir, filename);
    
    // Validate that the file exists and is a .json file
    if (!filename.endsWith('.json')) {
      return NextResponse.json(
        { error: 'Invalid file type' },
        { status: 400 }
      );
    }

    // Get the content from the request body
    const content = await request.text();
    
    // Validate JSON
    try {
      JSON.parse(content);
    } catch (e) {
      return NextResponse.json(
        { error: 'Invalid JSON content' },
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