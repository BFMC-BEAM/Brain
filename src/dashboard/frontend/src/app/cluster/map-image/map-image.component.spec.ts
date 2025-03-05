import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MapImageComponent } from './map-image.component';

describe('MapImageComponent', () => {
  let component: MapImageComponent;
  let fixture: ComponentFixture<MapImageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MapImageComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(MapImageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
