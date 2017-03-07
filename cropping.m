function [ cropped ]  = cropping(img, ave_intens)
img_size = size(img);

% we generate a matrix which contains the center of the image and raduis
circ_dim = [img_size(1)/2, img_size(2)/2, img_size(2)/2];   
[xx,yy] = ndgrid((1:img_size(1))-circ_dim(1),(1:img_size(2))-circ_dim(2));

%generate the mask that covers the original image
cover = uint8((xx.^2 + yy.^2) < circ_dim(3)^2);
% cover_1 = uint8(15*((xx.^2 + yy.^2) > circ_dim(3)^2));

%create new cropped image
cropped = uint8(ones(size(img)));

cropped(:,:,1) = img(:,:,1).*cover;
cropped(:,:,2) = img(:,:,2).*cover;
cropped(:,:,3) = img(:,:,3).*cover;

       
        for a = 1:img_size(2)        
                for b = 1:img_size(1)                        
                                if (a-img_size(1)/2)^2 + (b-img_size(2)/2)^2 -(img_size(1)/2)^2 > img_size(1)/2
%                                if uint((xx.^2 + yy.^2) > (circ_dim(3)^2))
                                    cropped(a,b,1) = ave_intens*ones(1, 'uint8');
                                    cropped(a,b,2) = ave_intens*ones(1, 'uint8');
                                    cropped(a,b,3) = ave_intens*ones(1, 'uint8');

                                end;
                            
                end;           
        end;



%multiply image by the cover to create a new image with all three colors

% cropped(:,:,1) = img(:,:,1).*cover_1;
% cropped(:,:,2) = img(:,:,2).*cover_1;
% cropped(:,:,3) = img(:,:,3).*cover_1;
imshow(cropped);