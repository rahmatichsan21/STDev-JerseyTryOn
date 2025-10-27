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
    
    // Jersey switching
    private string[] availableJerseys = { "none", "brighton", "brazil", "argentina", "germany" };
    private int currentJerseyIndex = 0;
    
    public override void _Ready()
    {
        // Dapatkan referensi ke node TextureRect
        textureRect = GetNode<TextureRect>("TextureRect");

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
                ProcessVideoFrame(packet);
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
            if (keyEvent.Keycode == Key.Key1)
            {
                ChangeJersey("none");
            }
            else if (keyEvent.Keycode == Key.Key2)
            {
                ChangeJersey("brighton");
            }
            else if (keyEvent.Keycode == Key.Key3)
            {
                ChangeJersey("brazil");
            }
            else if (keyEvent.Keycode == Key.Key4)
            {
                ChangeJersey("argentina");
            }
            else if (keyEvent.Keycode == Key.Key5)
            {
                ChangeJersey("germany");
            }
            else if (keyEvent.Keycode == Key.Space)
            {
                // Cycle through jerseys with spacebar
                currentJerseyIndex = (currentJerseyIndex + 1) % availableJerseys.Length;
                ChangeJersey(availableJerseys[currentJerseyIndex]);
            }
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

    private void ChangeJersey(string jerseyName)
    {
        if (socket.GetReadyState() == WebSocketPeer.State.Open)
        {
            var command = new Godot.Collections.Dictionary
            {
                {"type", "change_jersey"},
                {"jersey", jerseyName}
            };
            
            string jsonString = Json.Stringify(command);
            socket.SendText(jsonString);
            
            GD.Print($"Jersey changed to: {jerseyName}");
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