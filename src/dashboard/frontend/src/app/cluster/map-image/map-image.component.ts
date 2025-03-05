import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { Subscription } from 'rxjs';
import { WebSocketService } from '../../webSocket/web-socket.service';

@Component({
  selector: 'app-map-image',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './map-image.component.html',
  styleUrl: './map-image.component.css'
})
export class MapImageComponent {
  public image: string | undefined;
  public loading: boolean = true;
  public showImage: boolean = true; // Nueva variable para controlar la visibilidad
  private canvasSize: number[] = [450, 450];
  private cameraSubscription: Subscription | undefined;

  constructor(private webSocketService: WebSocketService) { }

  ngOnInit() {
    this.image = this.createBlackImage();

    this.cameraSubscription = this.webSocketService.receiveMap().subscribe(
      (message) => {
        this.image = `data:image/png;base64,${message.value}`;
        this.loading = false;
      },
      (error) => {
        this.image = this.createBlackImage();
        this.loading = true;
        console.error('Error receiving disk usage:', error);
      }
    );
  }

  ngOnDestroy() {
    if (this.cameraSubscription) {
      this.cameraSubscription.unsubscribe();
    }
    this.webSocketService.disconnectSocket();
  }

  createBlackImage(): string {
    const canvas = document.createElement('canvas');
    canvas.width = this.canvasSize[0];
    canvas.height = this.canvasSize[1];
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    return canvas.toDataURL('image/png');
  }

  toggleImageVisibility(): void {
    this.showImage = !this.showImage;
  }
}
