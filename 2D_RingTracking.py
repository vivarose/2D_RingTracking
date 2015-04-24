#created by Aaron Goldfain and Viva Horowitz
#April 24, 2015


import numpy as np

def RingTracking2D(data, n_centers, threshold = 0.5, draw_dots = False, draw_mean = False, return_dots = False, outfolder = None, static = True):
    # tracks a ring pattern in 2D. Locates particles using the Hough transfrom algorithm (center_find)
    # data is a stack of images to track particles in, stored as xyt hyperstack
    # n_centers is the number of ring patterns to track
    # theshold is the treshold used in center_find
    # draw_dots = True draws a label and a spot on each image from data where each center was found
    # draw_mean = True Uses the mean coordinate from each track as the position of the spot to be drawn
    # return_dots = True, retrun the images with the spots drawn on them
    # outfolder = location to save images with spots/labels
    # static = True uses first location of a particle to link tracks. Use False if the particle is moving
    #
    #
    # returns an array containing the coordinates of the ring centers at each frame
    # Also returns the images with a tracks labeled if return_dots = True

    #find centers
    from holopy.core.process.centerfinder import center_find
    coords = []
    print('finding particles')
    #loop through each frame
    for i in np.arange(0,data.shape[-1]):
        if i%50 == 0:
            print(i)
        coord = center_find(data[:,:,i], centers = n_centers, threshold = threshold)
        if n_centers ==1:
            coord = np.reshape(coord, (1,2))
        
        coords.append(coord)
    coords = np.array(coords)


    #link up particles between frames
    print('linking tracks')

    def get_distance(coords1, coords2):
        return np.sqrt( (coords1[0]-coords2[0])**2 + (coords1[1]-coords2[1])**2  )
        
    tracks = np.empty(coords.shape)

    #loop through each particle
    for particle in np.arange(0,coords.shape[1]):
        #set initial point
        tracks[0,particle,:] = coords[0,particle,:]
        known_coord = tracks[0,particle,:] #location of the particle for comparison
        
        #look in subsequent frames to find the closest particle to known_coord
        for frame in np.arange(1,coords.shape[0]):
            min_distance = 10000
            if not static:
                known_coord = tracks[frame-1,particle,:]
            #compare to all center locations
            for i in np.arange(0,coords.shape[1]):
                distance = get_distance(known_coord, coords[frame,i,:])
                if distance < min_distance:
                    min_distance = distance
                    closest_particle_index = i
            
            #set track coordinate     
            tracks[frame,particle,:] = coords[frame, closest_particle_index, :]

    if not draw_dots:
        return tracks

    #Draw a spot at each point that was found
    else:  
        print('drawing tracks')
        from Image import fromarray
        from ImageDraw import Draw

        #used for scaling images
        im_max = data.max()
        im_min = data.min()
        
        if return_dots:
            spot_images = []
        
        if draw_mean:
            tracksmean = np.mean(tracks,0)
        
        #loop through each frame
        for i in np.arange(0,data.shape[-1]):
            #convert to an RGB image
            spot_image = fromarray( 255*(data[:,:,i]-im_min)/(im_max-im_min) )
            spot_image = spot_image.convert('RGB')
            
            #draw spots and labels
            draw = Draw(spot_image)
            for j in np.arange(0,tracks.shape[1]):
                if draw_mean:
                    spot_coord = [np.round(tracksmean[j,1]), np.round(tracksmean[j,0])]                   
                else:
                    spot_coord = [np.round(tracks[i,j,1]), np.round(tracks[i,j,0])]        
                draw.point( (spot_coord[0],spot_coord[1]), (255,0,0))
                draw.text( (spot_coord[0]-10,spot_coord[1]-10), str(j),(255,0,0))
                       
            if return_dots:
                spot_images.append(spot_image)
            
            if outfolder != None:    
                spot_image.save(outfolder + 'spot_image' + str(i).zfill(5) +'.png')
                
        if return_dots:
            return [tracks, spot_images]
        else:
            return tracks


def track_intensity(data, track, use_spline = True, use_mean_after = None, r_avg = None, sub_px = 5, lower_threshold=-100):
    # gets the intensity in data at each frame at the point specified by track
    # use_spline = True uses a spline fit to interpolate sub-pixel points
    # use_mean_after uses a mean of all track points after frame number use_mean_after
    # r_avg: average the intensity within a radius of r_avg pixels
    # sub_px specifies how many subpixels to break the spline into when taking the circular average.
    
    from scipy.interpolate import RectBivariateSpline
    
    if use_mean_after != None:
        mean_track = np.mean(track[use_mean_after:,:],0)
        
    intensities =[]
    
    if use_spline:
        xs = np.arange(0,data.shape[0])
        ys = np.arange(0,data.shape[1])
    for i in np.arange(0,track.shape[0]):
        if i%100 ==0:
            print(i)
         
        #fit spline to data for interpolation    
        if use_spline:
            sp = RectBivariateSpline(xs, ys, data[:,:,i] )
        #get point to interpolate
        if use_mean_after!=None:
            point = np.array(mean_track)
        else:
            point = track[i,:]

        #interpolate for this point
        if use_spline:
            intensity = sp(point[0], point[1])[0,0]
        else:
            intensity = data[np.round(point[0]), np.round(point[1]), i]
        
        n_pts = 1
        #average over circular area    
        if r_avg != None and r_avg != 0:
            for radius in np.linspace(0,r_avg,sub_px*r_avg)[1:]:
                for theta in np.linspace(0, 2*np.pi, 2*np.pi*radius*sub_px)[1:]: 
                    x = point[0]+radius*np.cos(theta)
                    y = point[1]+radius*np.sin(theta)
                    if use_spline:
                        px_intensity = sp(x, y)[0,0]
                    else:
                        px_intensity = data[np.round(x), np.round(y), i]
                    
                    if px_intensity > lower_threshold:
                        intensity += px_intensity
                        n_pts +=1

                                
        intensities.append( intensity )
    
    intensities = np.array(intensities)/n_pts
    print(n_pts)
    
    return intensities


  
