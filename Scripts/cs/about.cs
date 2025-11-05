using Godot;
using System;

public partial class about : Control
{
    private Button back2Home;
    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
    {
        back2Home = GetNode<Button>("BackHome");
		back2Home.GetNode<Button>("Button").Text = back2Home.Text;
		back2Home.Pressed += OnBack2HomeButtonPressed;
    }

    // Called every frame. 'delta' is the elapsed time since the previous frame.
    public override void _Process(double delta)
    {
    }
    private void OnBack2HomeButtonPressed()
	{
		GetTree().ChangeSceneToFile("res://Scene/menu2.tscn");
	}
}
