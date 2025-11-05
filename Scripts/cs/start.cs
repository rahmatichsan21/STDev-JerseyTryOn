using Godot;
using System;

public partial class start : Control
{
	private Button back2Home;
	private Button capture;
	private Button startServer;
	private Button stopServer;
	
	// WebSocket untuk kamera
	private string websocketUrl = "ws://localhost:8765";
	private WebSocketPeer socket = new WebSocketPeer();
	private bool isConnected = false;

	// Variabel untuk video stream
	private Image videoImage = null;
	private ImageTexture videoTexture = null;
	private TextureRect cameraFrame;
	
	// UI untuk log messages
	private RichTextLabel logLabel;
	
	private bool isCapturing = false;
	private Image lastCapturedImage = null;
	
	// Variabel untuk mengelola server process
	private int serverProcessId = -1;
	private bool isServerRunning = false;
	
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		// Setup buttons
		back2Home = GetNode<Button>("BackHome");
		back2Home.GetNode<Button>("Button").Text = back2Home.Text;
		back2Home.Pressed += OnBack2HomeButtonPressed;

		capture = GetNode<Button>("Capture");
		capture.GetNode<Button>("Button").Text = capture.Text;
		capture.Pressed += OnCaptureButtonPressed;
		
		// Setup server control buttons
		startServer = GetNodeOrNull<Button>("StartServer");
		if (startServer != null)
		{
			startServer.GetNode<Button>("Button").Text = startServer.Text;
			startServer.Pressed += OnStartServerPressed;
		}
		
		stopServer = GetNodeOrNull<Button>("StopServer");
		if (stopServer != null)
		{
			stopServer.GetNode<Button>("Button").Text = stopServer.Text;
			stopServer.Pressed += OnStopServerPressed;
			stopServer.Disabled = true; // Disabled until server starts
		}
		
		// Setup camera frame
		cameraFrame = GetNode<TextureRect>("CameraFrame");
		
		// Setup log label (gunakan GetNodeOrNull untuk fallback jika belum dibuat)
		logLabel = GetNodeOrNull<RichTextLabel>("LogLabel");
		if (logLabel != null)
		{
			logLabel.BbcodeEnabled = true;
			UpdateLog("System ready. Press 'Start Server' to begin.");
		}
		
		// Don't auto-connect, wait for user to start server manually
		SetProcess(false);
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
		// Poll WebSocket untuk menerima data
		socket.Poll();

		var state = socket.GetReadyState();
		if (state == WebSocketPeer.State.Open)
		{
			if (!isConnected)
			{
				isConnected = true;
				GD.Print("Berhasil terhubung ke server!");
				UpdateLog("[color=green]✓ Connected to camera server[/color]");
			}

			// Proses semua paket yang masuk dari server
			while (socket.GetAvailablePacketCount() > 0)
			{
				byte[] packet = socket.GetPacket();
				
				// Check if it's a text message (JSON command response)
				string text = System.Text.Encoding.UTF8.GetString(packet);
				if (text.StartsWith("{"))
				{
					HandleTextMessage(text);
				}
				else
				{
					// It's a binary video frame
					ProcessVideoFrame(packet);
				}
			}
		}
		else if (state == WebSocketPeer.State.Closed)
		{
			if (isConnected)
			{
				isConnected = false;
				GD.Print("Koneksi ke server Python terputus.");
				UpdateLog("[color=red]✗ Connection lost[/color]");
			}
			SetProcess(false);
		}
	}

	private void OnBack2HomeButtonPressed()
	{
		// Tutup koneksi WebSocket sebelum pindah scene
		if (socket.GetReadyState() != WebSocketPeer.State.Closed)
		{
			socket.Close();
		}
		GetTree().ChangeSceneToFile("res://Scene/menu2.tscn");
	}
	
	private void OnCaptureButtonPressed()
	{
		CaptureAndSave();
	}
	
	private void OnStartServerPressed()
	{
		if (isServerRunning)
		{
			UpdateLog("[color=yellow]Server is already running![/color]");
			return;
		}
		
		UpdateLog("[color=cyan]Starting Python server...[/color]");
		GD.Print("Starting Python server...");
		
		// Get the path to the Python script
		string projectPath = ProjectSettings.GlobalizePath("res://");
		string scriptPath = System.IO.Path.Combine(projectPath, "Scripts", "Python", "jersey_filter_server_hybrid.py");
		
		// Create process start info
		var processInfo = new System.Diagnostics.ProcessStartInfo
		{
			FileName = "py",
			Arguments = $"-3.11 \"{scriptPath}\"",
			UseShellExecute = false,
			CreateNoWindow = false, // Show console window so user can see server logs
			WorkingDirectory = System.IO.Path.Combine(projectPath, "Scripts", "Python")
		};
		
		try
		{
			var process = System.Diagnostics.Process.Start(processInfo);
			if (process != null)
			{
				serverProcessId = process.Id;
				isServerRunning = true;
				
				// Update UI
				if (startServer != null) startServer.Disabled = true;
				if (stopServer != null) stopServer.Disabled = false;
				
				UpdateLog("[color=green]✓ Server started successfully![/color]");
				GD.Print($"Server process started with PID: {serverProcessId}");
				
				// Wait a bit for server to initialize, then connect
				GetTree().CreateTimer(2.0).Timeout += () => ConnectToServer();
			}
			else
			{
				UpdateLog("[color=red]✗ Failed to start server![/color]");
				GD.PrintErr("Failed to start server process");
			}
		}
		catch (System.Exception e)
		{
			UpdateLog($"[color=red]✗ Error starting server: {e.Message}[/color]");
			GD.PrintErr($"Error starting server: {e.Message}");
		}
	}
	
	private void OnStopServerPressed()
	{
		if (!isServerRunning || serverProcessId == -1)
		{
			UpdateLog("[color=yellow]No server is running![/color]");
			return;
		}
		
		UpdateLog("[color=yellow]Stopping server...[/color]");
		GD.Print("Stopping Python server...");
		
		try
		{
			var process = System.Diagnostics.Process.GetProcessById(serverProcessId);
			if (process != null && !process.HasExited)
			{
				// Close gracefully first
				process.CloseMainWindow();
				
				// Wait up to 3 seconds for graceful shutdown
				if (!process.WaitForExit(3000))
				{
					// Force kill if it doesn't close
					process.Kill();
					GD.Print("Server forced to stop");
				}
				
				process.Dispose();
			}
			
			// Close WebSocket connection
			if (socket.GetReadyState() != WebSocketPeer.State.Closed)
			{
				socket.Close();
			}
			
			isServerRunning = false;
			isConnected = false;
			serverProcessId = -1;
			SetProcess(false);
			
			// Update UI
			if (startServer != null) startServer.Disabled = false;
			if (stopServer != null) stopServer.Disabled = true;
			
			UpdateLog("[color=green]✓ Server stopped[/color]");
			GD.Print("Server stopped successfully");
		}
		catch (System.ArgumentException)
		{
			// Process doesn't exist anymore
			isServerRunning = false;
			serverProcessId = -1;
			
			if (startServer != null) startServer.Disabled = false;
			if (stopServer != null) stopServer.Disabled = true;
			
			UpdateLog("[color=yellow]Server was already stopped[/color]");
		}
		catch (System.Exception e)
		{
			UpdateLog($"[color=red]✗ Error stopping server: {e.Message}[/color]");
			GD.PrintErr($"Error stopping server: {e.Message}");
		}
	}
	
	private void ConnectToServer()
	{
		GD.Print("Attempting to connect to server...");
		UpdateLog("Connecting to camera server...");
		
		Error err = socket.ConnectToUrl(websocketUrl);
		if (err != Error.Ok)
		{
			GD.PrintErr("Error: Cannot connect to server.");
			UpdateLog("[color=red]Error: Cannot connect to server![/color]");
			UpdateLog("[color=yellow]Make sure server is running...[/color]");
			SetProcess(false);
		}
		else
		{
			SetProcess(true);
		}
	}
	
	private void HandleTextMessage(string jsonText)
	{
		try
		{
			var json = Json.ParseString(jsonText);
			if (json.AsGodotDictionary().ContainsKey("type"))
			{
				string type = json.AsGodotDictionary()["type"].AsString();
				
				if (type == "capture_ready")
				{
					GD.Print("Capture ready! Saving image...");
					UpdateLog("[color=yellow]Capture ready! Saving image...[/color]");
					SaveCapturedImage();
				}
				else if (type == "processing_complete")
				{
					bool success = json.AsGodotDictionary().ContainsKey("success") && 
								   (bool)json.AsGodotDictionary()["success"];
					if (success && json.AsGodotDictionary().ContainsKey("output_path"))
					{
						string outputPath = json.AsGodotDictionary()["output_path"].AsString();
						GD.Print($"✓ Jersey application complete!");
						GD.Print($"   Result saved to: {outputPath}");
						UpdateLog("[color=green]✓ Jersey application complete![/color]");
						
						// Optionally load and display the processed image
						LoadProcessedImage(outputPath);
					}
					else
					{
						GD.PrintErr("✗ Jersey processing failed");
						UpdateLog("[color=red]✗ Jersey processing failed[/color]");
					}
				}
				else if (type == "error")
				{
					string errorMsg = json.AsGodotDictionary().ContainsKey("message") ? 
									 json.AsGodotDictionary()["message"].AsString() : "Unknown error";
					GD.PrintErr($"Python error: {errorMsg}");
					UpdateLog($"[color=red]Error: {errorMsg}[/color]");
				}
			}
		}
		catch (System.Exception e)
		{
			GD.PrintErr($"Error parsing command: {e.Message}");
		}
	}
	
	private void ProcessVideoFrame(byte[] byteArray)
	{
		if (videoImage == null)
		{
			// Inisialisasi frame pertama
			videoImage = new Image();
			Error err = videoImage.LoadJpgFromBuffer(byteArray);

			if (err == Error.Ok)
			{
				// Buat ImageTexture dari Image
				videoTexture = ImageTexture.CreateFromImage(videoImage);
				// Tetapkan tekstur ke CameraFrame untuk menampilkannya
				cameraFrame.Texture = videoTexture;
				GD.Print("Frame pertama diterima, kamera aktif.");
				UpdateLog("[color=cyan]Camera active - streaming...[/color]");
			}
			else
			{
				GD.PrintErr("Error saat mendekode frame JPEG pertama.");
				videoImage = null;
			}
		}
		else
		{
			// Update frame berikutnya
			Error err = videoImage.LoadJpgFromBuffer(byteArray);

			if (err == Error.Ok)
			{
				// Update tekstur yang sudah ada (lebih cepat)
				videoTexture.Update(videoImage);
			}
			else
			{
				GD.PrintErr("Error saat mendekode frame JPEG.");
			}
		}
	}
	
	private void CaptureAndSave()
	{
		if (socket.GetReadyState() == WebSocketPeer.State.Open)
		{
			if (videoImage != null)
			{
				// Store the last frame
				lastCapturedImage = videoImage;
				isCapturing = true;
				
				// Tell Python to pause streaming
				var command = new Godot.Collections.Dictionary
				{
					{"type", "save_capture"}
				};
				string jsonString = Json.Stringify(command);
				socket.SendText(jsonString);
				
				GD.Print("Capturing frame...");
				UpdateLog("[color=yellow]📸 Capturing frame...[/color]");
			}
			else
			{
				GD.PrintErr("No video frame available to capture!");
			}
		}
		else
		{
			GD.PrintErr("Cannot capture: WebSocket not open.");
		}
	}
	
	private void SaveCapturedImage()
	{
		if (lastCapturedImage != null)
		{
			// Create a timestamp for unique filename
			string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
			string filename = $"user://captured_{timestamp}.jpg";
			
			// Save the image as JPEG
			Error err = lastCapturedImage.SaveJpg(filename, 0.95f);
			
			if (err == Error.Ok)
			{
				// Get the actual file path
				string realPath = ProjectSettings.GlobalizePath(filename);
				GD.Print($"✓ Image saved successfully to: {realPath}");
				UpdateLog($"[color=green]✓ Image saved[/color]");
				
				// Send processing request to Python
				ProcessImageWithJersey(realPath, timestamp);
			}
			else
			{
				GD.PrintErr($"✗ Failed to save image. Error: {err}");
				UpdateLog($"[color=red]✗ Failed to save image[/color]");
			}
			
			isCapturing = false;
			
			// Auto-resume streaming after capture
			ResumeStreaming();
		}
	}
	
	private void ProcessImageWithJersey(string imagePath, string timestamp)
	{
		if (socket.GetReadyState() == WebSocketPeer.State.Open)
		{
			// Prepare output path
			string outputFilename = $"user://processed_{timestamp}.png";
			string outputPath = ProjectSettings.GlobalizePath(outputFilename);
			
			var command = new Godot.Collections.Dictionary
			{
				{"type", "process_image"},
				{"image_path", imagePath},
				{"jersey", "Brighton Home.png"},
				{"output_path", outputPath}
			};
			
			string jsonString = Json.Stringify(command);
			socket.SendText(jsonString);
			
			GD.Print($"📤 Sent image for processing: {imagePath}");
			GD.Print($"   Output will be saved to: {outputPath}");
			UpdateLog("[color=cyan]📤 Processing with jersey filter...[/color]");
		}
		else
		{
			GD.PrintErr("Cannot send processing request: WebSocket not open.");
		}
	}
	
	private void LoadProcessedImage(string imagePath)
	{
		var processedImage = Image.LoadFromFile(imagePath);
		if (processedImage != null)
		{
			GD.Print($"✓ Processed image loaded successfully");
			// Optionally display the processed result in CameraFrame
			// cameraFrame.Texture = ImageTexture.CreateFromImage(processedImage);
		}
		else
		{
			GD.PrintErr($"Failed to load processed image from: {imagePath}");
		}
	}
	
	private void ResumeStreaming()
	{
		if (socket.GetReadyState() == WebSocketPeer.State.Open)
		{
			var command = new Godot.Collections.Dictionary
			{
				{"type", "resume"}
			};
			string jsonString = Json.Stringify(command);
			socket.SendText(jsonString);
			
			isCapturing = false;
			GD.Print("Streaming resumed.");
			UpdateLog("[color=cyan]Streaming resumed[/color]");
		}
	}
	
	// Helper method untuk update log UI
	private void UpdateLog(string message)
	{
		if (logLabel != null)
		{
			// Add timestamp
			string timestamp = System.DateTime.Now.ToString("HH:mm:ss");
			logLabel.Text = $"[{timestamp}] {message}";
			
			// Optional: Keep history (uncomment if you want to show multiple lines)
			// logLabel.Text += $"\n[{timestamp}] {message}";
		}
	}

	public override void _ExitTree()
	{
		// Stop server if it's still running
		if (isServerRunning && serverProcessId != -1)
		{
			try
			{
				var process = System.Diagnostics.Process.GetProcessById(serverProcessId);
				if (process != null && !process.HasExited)
				{
					process.Kill();
					process.Dispose();
					GD.Print("Server process terminated on exit");
				}
			}
			catch (System.Exception e)
			{
				GD.PrintErr($"Error terminating server on exit: {e.Message}");
			}
		}
		
		// Tutup koneksi WebSocket dengan bersih saat keluar
		if (socket.GetReadyState() != WebSocketPeer.State.Closed)
		{
			socket.Close();
		}
	}
}
