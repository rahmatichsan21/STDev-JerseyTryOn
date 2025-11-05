using Godot;

public partial class Main : Control
{
	private string websocketUrl = "ws://localhost:8765";
	private WebSocketPeer socket = new WebSocketPeer();
	private bool isConnected = false;

	// Variabel untuk menyimpan referensi ke Image dan ImageTexture
	// Ini adalah kunci optimisasi untuk performa yang lancar
	private Image videoImage = null;
	private ImageTexture videoTexture = null;

	private TextureRect textureRect;
	private Button captureButton;
	
	private bool isCapturing = false;
	private Image lastCapturedImage = null;
	
	public override void _Ready()
	{
		// Dapatkan referensi ke node TextureRect
		textureRect = GetNode<TextureRect>("TextureRect");
		
		// Get reference to capture button (adjust the path to match your scene structure)
		// Common button paths: "CaptureButton", "Button", "Panel/CaptureButton", etc.
		captureButton = GetNodeOrNull<Button>("CaptureButton");
		
		if (captureButton != null)
		{
			captureButton.Pressed += OnCaptureButtonPressed;
			GD.Print("Capture button connected!");
		}
		else
		{
			GD.PrintErr("Warning: CaptureButton not found. Trying alternative paths...");
			// Try alternative common paths
			captureButton = GetNodeOrNull<Button>("Button");
			if (captureButton == null)
				captureButton = GetNodeOrNull<Button>("Panel/Button");
			if (captureButton == null)
				captureButton = GetNodeOrNull<Button>("UI/CaptureButton");
			
			if (captureButton != null)
			{
				captureButton.Pressed += OnCaptureButtonPressed;
				GD.Print($"Capture button found and connected!");
			}
			else
			{
				GD.Print("No button found. Using C key as fallback.");
			}
		}

		GD.Print("Mencoba terhubung ke server Python...");
		Error err = socket.ConnectToUrl(websocketUrl);
		if (err!= Error.Ok)
		{
			GD.PrintErr("Error: Tidak dapat terhubung ke server Python.");
			SetProcess(false); // Matikan _Process loop jika koneksi gagal
		}
		else
		{
			SetProcess(true);
		}
	}

	public override void _Process(double delta)
	{
		// poll() harus dipanggil setiap frame untuk memproses data jaringan
		socket.Poll();

		var state = socket.GetReadyState();
		if (state == WebSocketPeer.State.Open)
		{
			if (!isConnected)
			{
				isConnected = true;
				GD.Print("Berhasil terhubung!");
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
			}
			SetProcess(false);
		}
	}

	public override void _Input(InputEvent @event)
	{
		if (@event is InputEventKey keyEvent && keyEvent.Pressed)
		{
			if (keyEvent.Keycode == Key.C)
			{
				// Press C to capture and save current frame (fallback if no button)
				CaptureAndSave();
			}
			else if (keyEvent.Keycode == Key.R)
			{
				// Press R to resume streaming after capture
				ResumeStreaming();
			}
		}
	}
	
	// Button handler for capture
	private void OnCaptureButtonPressed()
	{
		CaptureAndSave();
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
						
						// Optionally load and display the processed image
						LoadProcessedImage(outputPath);
					}
					else
					{
						GD.PrintErr("✗ Jersey processing failed");
					}
				}
				else if (type == "error")
				{
					string errorMsg = json.AsGodotDictionary().ContainsKey("message") ? 
									 json.AsGodotDictionary()["message"].AsString() : "Unknown error";
					GD.PrintErr($"Python error: {errorMsg}");
				}
			}
		}
		catch (System.Exception e)
		{
			GD.PrintErr($"Error parsing command: {e.Message}");
		}
	}
	
	// Load and optionally display the processed image
	private void LoadProcessedImage(string imagePath)
	{
		var processedImage = Image.LoadFromFile(imagePath);
		if (processedImage != null)
		{
			GD.Print($"✓ Processed image loaded successfully");
			// You can display it somewhere if needed
			// For example, update the texture to show the processed result:
			// lastCapturedImage = processedImage;
			// textureRect.Texture = ImageTexture.CreateFromImage(processedImage);
		}
		else
		{
			GD.PrintErr($"Failed to load processed image from: {imagePath}");
		}
	}
	
	// Fungsi ini memproses data byte gambar dan menampilkannya
	private void ProcessVideoFrame(byte[] byteArray)
	{
		if (videoImage == null)
		{
			// --- PENANGANAN FRAME PERTAMA (INisialisasi) ---
			videoImage = new Image();
			Error err = videoImage.LoadJpgFromBuffer(byteArray);

			if (err == Error.Ok)
			{
				// Buat ImageTexture dari Image
				videoTexture = ImageTexture.CreateFromImage(videoImage);
				// Tetapkan tekstur ke TextureRect untuk menampilkannya
				textureRect.Texture = videoTexture;
				GD.Print("Frame pertama diterima, tekstur diinisialisasi.");
			}
			else
			{
				GD.PrintErr("Error saat mendekode frame JPEG pertama.");
				videoImage = null; // Reset agar bisa mencoba lagi
			}
		}
		else
		{
			// --- PENANGANAN FRAME BERIKUTNYA (OPTIMISASI) ---
			Error err = videoImage.LoadJpgFromBuffer(byteArray);

			if (err == Error.Ok)
			{
				// Perbarui ImageTexture yang sudah ada dengan data Image baru.
				// Ini jauh lebih cepat daripada membuat tekstur baru setiap frame.[1, 2]
				videoTexture.Update(videoImage);
			}
			else
			{
				GD.PrintErr("Error saat mendekode frame JPEG.");
			}
		}
	}

	// Capture current frame and save it
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
	
	// Save the captured image to disk
	private void SaveCapturedImage()
	{
		if (lastCapturedImage != null)
		{
			// Create a timestamp for unique filename
			string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
			string filename = $"user://captured_{timestamp}.jpg";
			
			// Save the image as JPEG (avoids PNG format issues)
			Error err = lastCapturedImage.SaveJpg(filename, 0.95f);
			
			if (err == Error.Ok)
			{
				// Get the actual file path (Godot's user:// maps to a real directory)
				string realPath = ProjectSettings.GlobalizePath(filename);
				GD.Print($"✓ Image saved successfully to: {realPath}");
				
				// Send processing request to Python
				ProcessImageWithJersey(realPath, timestamp);
			}
			else
			{
				GD.PrintErr($"✗ Failed to save image. Error: {err}");
			}
			
			isCapturing = false;
			
			// Auto-resume streaming after capture
			ResumeStreaming();
		}
	}
	
	// Send image to Python for jersey processing
	private void ProcessImageWithJersey(string imagePath, string timestamp)
	{
		if (socket.GetReadyState() == WebSocketPeer.State.Open)
		{
			// Prepare output path (still PNG for final result)
			string outputFilename = $"user://processed_{timestamp}.png";
			string outputPath = ProjectSettings.GlobalizePath(outputFilename);
			
			var command = new Godot.Collections.Dictionary
			{
				{"type", "process_image"},
				{"image_path", imagePath},  // JPEG file from Godot
				{"jersey", "Brighton Home.png"},  // Using Brighton Home jersey
				{"output_path", outputPath}
			};
			
			string jsonString = Json.Stringify(command);
			socket.SendText(jsonString);
			
			GD.Print($"📤 Sent image for processing: {imagePath}");
			GD.Print($"   Output will be saved to: {outputPath}");
		}
		else
		{
			GD.PrintErr("Cannot send processing request: WebSocket not open.");
		}
	}
	
	// Resume video streaming
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
		}
	}

	public override void _ExitTree()
	{
		// Pastikan koneksi ditutup dengan bersih saat aplikasi keluar
		if (socket.GetReadyState() != WebSocketPeer.State.Closed)
		{
			socket.Close();
		}
	}
}
