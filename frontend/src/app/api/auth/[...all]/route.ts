import { NextRequest, NextResponse } from "next/server"

// Force dynamic rendering to prevent build-time evaluation
export const dynamic = "force-dynamic"
export const runtime = "nodejs"

// Simple handler that returns 404 since auth is not being used
export async function GET(request: NextRequest) {
    return NextResponse.json(
        { error: "Auth endpoint not configured" },
        { status: 404 }
    )
}

export async function POST(request: NextRequest) {
    return NextResponse.json(
        { error: "Auth endpoint not configured" },
        { status: 404 }
    )
}
