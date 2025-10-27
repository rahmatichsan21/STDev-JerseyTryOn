using Godot;

public partial class MainMenu : Control
{
    // Default path to your Start button in the scene tree.
    // You can change this in the Inspector after attaching the script (no need to edit .tscn manually).
    [Export]
    public NodePath StartButtonPath { get; set; } = "Panel/TextureButton";

    private TextureButton _startButton;

    public override void _Ready()
    {
        _startButton = GetNodeOrNull<TextureButton>(StartButtonPath);
        if (_startButton != null)
        {
            _startButton.Pressed += OnStartPressed;
        }
        else
        {
            GD.PushError($"Start button not found at path: {StartButtonPath}");
        }
    }

    private void OnStartPressed()
    {
        var err = GetTree().ChangeSceneToFile("res://Scene/main.tscn");
        if (err != Error.Ok)
        {
            GD.PushError($"Failed to change scene to res://Scene/main.tscn: {err}");
        }
    }
}
