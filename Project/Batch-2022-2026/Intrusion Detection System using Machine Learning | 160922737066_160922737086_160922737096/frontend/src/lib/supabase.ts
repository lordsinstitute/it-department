import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = "https://jnpvoavhsjrhnuljklbm.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpucHZvYXZoc2pyaG51bGprbGJtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQyNjk3OTgsImV4cCI6MjA4OTg0NTc5OH0.GM3Sk0c9Eh_b4vuEvW1jO1xHjiDfnUpKt7TH9Qcye08";

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
