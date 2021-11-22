function showInput(row,col,data)
    tiledlayout(row,col, 'Padding', 'none', 'TileSpacing', 'none'); 
    for i=1:row
        for j=1:col
            %subplot(row,col,i*col+j)
            %sprintf('i= %d j=%d', i,j)
            nexttile;
            imshow(reshape(data(:,i*col+j),[28,28]));
        end
    end