R = 800;
s_r = 12;
s_f = 17;
OG = 300;
t_1 = OG/s_f;

%Plot the rabbit's and the fox's paths from the initial time to time t1
tspan0 = 1:0.0001:t_1;
r0 = [-R*sin(s_r*tspan0/R); R*cos(s_r*tspan0/R)];
plot(r0(1,:),r0(2,:),'green',LineWidth=3);

hold on

x = 0;
y = linspace(0,300,10000);
plot(x,y,'-o');

TF = R*pi/3/12;
tspan = t_1:0.0001:TF;%Discretise the time

%The real-time position of the rabbit
r = [-R*sin(s_r*tspan/R); R*cos(s_r*tspan/R)];

%Plot the rabbit's path after time t_1
plot(r(1,:),r(2,:),'-o');

%Define the ODE function as specified in equation (4)
odefun = @(t,z) [(s_f*(rpos(t,1,1)-z(1))/sqrt((rpos(t,1,1)-z(1))^2+(rpos(t,2,1)-z(2))^2));(s_f*(rpos(t,2,1)-z(2))/sqrt((rpos(t,1,1)-z(1))^2+(rpos(t,2,1)-z(2))^2))];

%Use ode45 to return an array of solutions (z: the fox's position) and a
%column vector of evaluation points (t: time)
%The initial point is G(0,300)
[t,z] = ode45(odefun,tspan,[0 300]);

%Determine if the fox can catch the rabbit
for i = 1:size(t,1)
    catch_distance = sqrt((r(1,i) - z(i,1))^2+(r(2,i) - z(i,2))^2);
    if catch_distance < 0.1
        disp("Caught at time t2");
        disp("Time:");
        disp([t_1 t(i)]);
        disp("Place:");
        disp([r(1,i) r(2,i)]);
        return;
    end
    if cantsee(r(1,i),r(2,i),z(i,1),z(i,2))
        break;
    end
end
t_2 = t(i);

%Plot the fox's path from time t1 to t2
plot(z(1:i,1),z(1:i,2),'-o');

if ~cantsee(r(1,i),r(2,i),z(i,1),z(i,2))
    disp("Can't catch it...");
    disp("Time:");
    disp([t_1 t_2]);
    disp("Fox's position:");
    disp([z(i,1) z(i,2)]);
    return;
end

%The time for the fox to travel from the position where it cannot see the
%rabbit to point A
t_3 = t_2 + sqrt((350+z(i,1))^2 + (620-z(i,2))^2)/s_f;

tspan2 = t_3:0.0001:TF;
r2 = [-R*sin(s_r*tspan2/R); R*cos(s_r*tspan2/R)];

%ode45 with initial position set to point A
[t2,z2] = ode45(odefun,tspan2,[-350 620]);

%Plot fox's path from time t2 to t3
z_x = [z(i,1) z2(1,1)];
z_y = [z(i,2) z2(1,2)];
plot(z_x,z_y,LineWidth=3,Color='magenta');
plot(z2(:,1),z2(:,2),'-o');
hold off

%Determine if the fox can catch the rabbit after the fox passes A
for i = 1:size(t2,1)
    catch_distance = sqrt((r2(1,i) - z2(i,1))^2+(r2(2,i) - z2(i,2))^2);
    if catch_distance < 0.1
        disp("Caught at time t4");
        disp("Time:");
        disp([t_1 t_2 t_3 t2(i)]);
        disp("Place:");
        disp([r2(1,i) r2(2,i)]);
        return;
    end
end

fox_total_distance = s_f*t2(i);

disp("Can't catch it...");
disp("Time:");
disp([t_1 t_2 t_3 TF]);
disp("Fox's position:");
disp([z2(i,1) z2(i,2)]);
disp("Fox's distance:");
disp(fox_total_distance);