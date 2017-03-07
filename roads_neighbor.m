function [] = roads_neighbor(x)

% converting to hsv 
hsv = rgb2hsv(x);             
satur = 100.*hsv(:,:,2);    
value = 100.*hsv(:,:,3);
%setting the limits of saturation and value planes
s_upper = 25;
s_lower= 5;
v_upper = 65;
v_lower = 35;

%threshold we set up for whole images
threshold = 15000;
queue = CQueue();
% sizes
row = size(x,1);
column = size(x,2);
% creating a matrix which track pixels which were already processed
done_pixel = zeros(row+2,column+2);
done_pixel(:,1)=1;
done_pixel(1,:)=1;
done_pixel(end,:)=1;
done_pixel(:,end)=1;
% creatin a final matrix with river
final_matrix = zeros(row, column);



% nested for loop which go through aaaalll pixels of the image
for i = 1:row
    for j = 1:column
%       if ( lower_r < x(i,j,1) <upper_r) && ( lower_g < x(i,j,2) < upper_g) && (x(i,j,3)>lower_b && x(i,j,3) <upper_b)
%        checking whether the given pixels in the range of color
         if (satur(i,j)>s_lower) && (satur(i,j)<s_upper) && (value(i,j)<v_upper) && (value(i,j)>v_lower)
%           creating a list that will save all pixels of interest
            clear list;
            list(1,1) = 1;
            list_index = 0;
%           we escape the loop and go to the next pixels if the current pixel was proccesses alraedy
            if done_pixel(i+1,j+1) == 1
                continue;
            end
%           if not, we push the current pixel to the queue. It is done in
%           order to track all the pixels which are connected the current
%           pixel [i,j]
            queue.push([i,j])
%           we alse mark the current pixel as proccessed
            done_pixel(i+1,j+1) = 1;
%           iterate while the connection between pixels is not interrupted
            while(~queue.isempty())
%               taking out the first pixel because it is already processed
%               and taking its values 
                ab=queue.pop();  
                a=ab(1);
                b=ab(2);
%               now, we iterate through all adjacent pixels to check if they fall under condition
                for y = a-1:a+1
                   for z = b-1:b+1
                       %escape if it the current pixel
                       if (y == a && z ==b)
                           continue;
                       end
%                        #esacpe is it processed already
                       if done_pixel(y+1,z+1) == 1
                           continue;
                       end
%                        mark the current pixel as processed
                       done_pixel(y+1,z+1) = 1;

%                      now, check if the current pixels fall under
%                      condition and push to the queue, update the list of
%                      adjecent pixels to save its coordinates.
                       if (satur(y,z)<s_upper) && (value(y,z)<v_upper) && (value(y,z)>v_lower)
                            queue.push([y,z])
                            list_index = list_index +1;
                            list(list_index,1) = y;
                            list(list_index,2) = z;                                
                       end 
                   
                   end
                end
            end
            %if the number of pixels in the list is bigger than the
            %threshold we write them to our binary matrix
            if (list_index > threshold)
                for m=1:list_index
                    final_matrix(list(m,1),list(m,2)) = 1;
                end
            end
        end
    end
end 
%finally we calculate the total area of the river and show as binary image.
imshow(final_matrix)
disp('The total area of roads is:')
total_area  = bwarea(final_matrix);
disp(total_area);